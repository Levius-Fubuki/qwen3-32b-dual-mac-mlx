"""Standalone weighted two-rank pipeline adapter for Qwen3."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import (
    ModelArgs as UpstreamModelArgs,
    Qwen3Model as UpstreamQwen3Model,
)


@dataclass(frozen=True)
class PipelinePartition:
    rank: int
    world_size: int
    stage_index: int
    start: int
    end: int


def partition_layers(
    num_layers: int,
    rank: int,
    world_size: int,
    stage_layers: Optional[Sequence[int]] = None,
) -> PipelinePartition:
    """Map forward-order stages onto reverse-order distributed ranks."""
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    if type(world_size) is not int or world_size not in (1, 2):
        raise ValueError("world_size must be 1 or 2")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be in range(world_size)")

    if stage_layers is None:
        layers_per_stage, extra = divmod(num_layers, world_size)
        sizes = tuple(
            layers_per_stage + (stage_index < extra)
            for stage_index in range(world_size)
        )
    else:
        if not isinstance(stage_layers, Sequence):
            raise ValueError("stage_layers must be a sequence")
        sizes = tuple(stage_layers)
        if len(sizes) != world_size:
            raise ValueError("stage_layers must contain one entry per rank")

    if any(
        type(size) is not int or size <= 0
        for size in sizes
    ):
        raise ValueError("stage layer counts must be positive integers")
    if sum(sizes) != num_layers:
        raise ValueError("stage layer counts must sum to num_layers")

    stage_index = world_size - 1 - rank
    start = sum(sizes[:stage_index])
    return PipelinePartition(
        rank=rank,
        world_size=world_size,
        stage_index=stage_index,
        start=start,
        end=start + sizes[stage_index],
    )


@dataclass
class ModelArgs(UpstreamModelArgs):
    pipeline_stage_layers: Optional[List[int]] = None


class Qwen3PipelineModel(UpstreamQwen3Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self._pipeline_signature = None

    def pipeline(self, group) -> None:
        rank = group.rank()
        world_size = group.size()
        if type(world_size) is not int or world_size != 2:
            raise ValueError("Qwen3 pipeline execution requires exactly two ranks")

        stage_layers = self.args.pipeline_stage_layers
        split = None if stage_layers is None else tuple(stage_layers)
        signature = (rank, world_size, split)
        if self._pipeline_signature is not None:
            if self._pipeline_signature == signature:
                return
            raise RuntimeError("model is already partitioned with a different rank or split")

        partition = partition_layers(
            self.num_hidden_layers,
            rank=rank,
            world_size=world_size,
            stage_layers=stage_layers,
        )
        self.pipeline_rank = rank
        self.pipeline_size = world_size
        self.start_idx = partition.start
        self.end_idx = partition.end
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self._pipeline_signature = signature

    @property
    def pipeline_layers(self):
        return [
            layer
            for layer in self.layers[self.start_idx : self.end_idx]
            if layer is not None
        ]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if not self.pipeline_layers:
            raise RuntimeError("pipeline stage has no local layers")

        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        cache = [None] * len(self.pipeline_layers) if cache is None else cache
        try:
            cache_length = len(cache)
        except TypeError as exc:
            raise ValueError("cache must be a sequence") from exc
        if cache_length != len(self.pipeline_layers):
            raise ValueError("cache must contain one entry per local pipeline layer")
        populated = [layer_cache is not None for layer_cache in cache]
        if any(populated) and not all(populated):
            raise ValueError("cache entries must be all None or all populated")
        mask = create_attention_mask(h, cache[0])
        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, self.pipeline_rank + 1)
        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache)
        if self.pipeline_rank != 0:
            sent_h = mx.distributed.send(h, self.pipeline_rank - 1)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, sent_h)
            h = sent_h
        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3PipelineModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

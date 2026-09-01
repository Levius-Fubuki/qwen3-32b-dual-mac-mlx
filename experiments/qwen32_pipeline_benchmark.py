from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generation-tokens", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=64)
    parser.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    args = parser.parse_args()

    mx.random.seed(0)
    group = mx.distributed.init()
    model, tokenizer, config = sharded_load(
        args.model,
        pipeline_group=group,
        tensor_group=None,
        return_config=True,
    )
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab_size, (args.prompt_tokens,)).tolist()

    started = time.perf_counter()
    response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.generation_tokens,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_group_size=64,
        quantized_kv_start=args.quantized_kv_start,
    ):
        pass
    elapsed = time.perf_counter() - started
    if response is None:
        raise RuntimeError("generation produced no response")
    if group.rank() == 0:
        print(
            json.dumps(
                {
                    "elapsed_s": elapsed,
                    "generation_tokens": response.generation_tokens,
                    "generation_tps": response.generation_tps,
                    "kv_bits": args.kv_bits,
                    "peak_memory_gb": response.peak_memory,
                    "prompt_tokens": response.prompt_tokens,
                    "prompt_tps": response.prompt_tps,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

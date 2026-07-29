#!/usr/bin/env python3
"""Measure InstantTensor's per-tensor path on one real K3 quant shard."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from instanttensor import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    process_group = None
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", device_id=local_rank)
        process_group = torch.distributed.group.WORLD
    started = time.perf_counter()
    tensors = []
    tensor_names = []
    cumulative_bytes = []
    total_bytes = 0
    opened = time.perf_counter()
    with safe_open(
        args.filename,
        framework="pt",
        device=local_rank,
        process_group=process_group,
        copy=args.copy,
    ) as weights:
        opened = time.perf_counter()
        for name, source in weights.tensors():
            destination = torch.empty_like(source)
            destination.copy_(source)
            total_bytes += source.numel() * source.element_size()
            if args.retain or args.verify:
                tensors.append(destination)
                tensor_names.append(name)
                cumulative_bytes.append(total_bytes)
    finished = time.perf_counter()
    verified = None
    if args.verify:
        if len(args.filename) != 1 or process_group is not None:
            raise ValueError("--verify currently requires one file and one rank")
        from safetensors import safe_open as reference_safe_open

        boundary = next(
            (i for i, offset in enumerate(cumulative_bytes) if offset >= 1 << 32),
            len(tensors) - 1,
        )
        sample_indices = sorted(
            {
                0,
                1,
                len(tensors) // 4,
                len(tensors) // 2,
                3 * len(tensors) // 4,
                max(0, boundary - 1),
                boundary,
                min(len(tensors) - 1, boundary + 1),
                len(tensors) - 2,
                len(tensors) - 1,
            }
        )
        verified = []
        with reference_safe_open(args.filename[0], framework="pt", device="cpu") as reference:
            for index in sample_indices:
                name = tensor_names[index]
                if not torch.equal(tensors[index].cpu(), reference.get_tensor(name)):
                    raise AssertionError(f"content mismatch at tensor {index}: {name}")
                verified.append({"index": index, "name": name})
    if process_group is not None:
        torch.distributed.barrier()
    if rank == 0:
        print(
            json.dumps(
            {
                "copy": args.copy,
                "retain": args.retain,
                "tensors": len(tensors) if args.retain else None,
                "bytes": total_bytes,
                "open_s": opened - started,
                "iterate_and_close_s": finished - opened,
                "total_s": finished - started,
                "effective_GBps": total_bytes / 1e9 / (finished - started),
                "world_size": (
                    torch.distributed.get_world_size()
                    if process_group is not None
                    else 1
                ),
                "verified": verified,
            },
            sort_keys=True,
            )
        )
    if process_group is not None:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

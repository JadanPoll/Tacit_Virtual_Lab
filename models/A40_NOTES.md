# A40-inspired C simulator: model notes

The [single-file C model](a40_model.c) is a revised version of an approximately 1,900-line A40-oriented simulator originally supplied as a PDF. It preserves structural ideas including SMs, warps, simplified issue behavior, caches, DRAM timing, coalescing and divergence.

## Experiments

```sh
gcc -std=c11 -O2 -Wall -Wextra -Werror models/a40_model.c -lm -o a40_model
./a40_model --sms 1 --warps 1 --instructions 48 --workload memory --pattern 0 --quiet
./a40_model --sms 1 --warps 1 --instructions 48 --workload memory --pattern 2 --quiet
./a40_model --probe --base 96 --stride 4
./a40_model --branch-probe --predicate 2 --mask 0xffffffff
./a40_model --cache-seq 0,32768,65536,98304,0,131072,32768,0
python3 models/test_a40.py
```

The six memory patterns are contiguous, stride 2, stride 32, scattered, broadcast and mixed. Parameters include `--cycles`, `--sms`, `--warps` and `--instructions`. Reaching a cycle limit is reported as a failed rather than completed experiment.

## Findings from the original source

The original program crashed on its default process stack. With an unlimited stack, short light-workload simulations progressed slowly. Workload setup repeatedly overwrote an instruction slot that the fetch path did not consume.

The revised source changes memory allocation, makes workload scenarios selectable, corrects misleading sector summaries, frees allocations and labels modeled performance quantities rather than presenting unsupported hardware speedup claims.

## Tests performed during the revision

- Strict `-Wall -Wextra -Wpedantic -Werror` compilation passed.
- Seven test families passed: scenario comparisons, randomized address/mask probes, eight branch predicates, cache LRU conflicts, mixed workloads and bounded termination.
- An AddressSanitizer/UBSan run with 2 SMs × 3 warps completed without reported errors.
- A short 84-SM run completed 2,688 modeled compute instructions.

These results verify behavior under the tested model, not agreement with actual NVIDIA hardware timings.

## Model fidelity

Instruction arithmetic is not executed as a CUDA program. Scheduling, cache and DRAM timing, and branch reconvergence are simplified. A 70–85% timing-correlation claim in the original PDF had no reproducible comparison data. Cycle count, IPC, utilization and speedup should therefore be read as outputs of the stated model rather than measured A40 values.

On Ampere, crossing an aligned 128-byte cache-line boundary does not by itself imply poor coalescing; counting the requested 32-byte sectors is more informative. See [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

The browser GPU problems currently use a simpler JavaScript adaptation. They do not yet execute the C model via WebAssembly.

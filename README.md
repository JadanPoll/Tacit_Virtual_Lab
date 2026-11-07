# Tacit Virtual Lab

The public practice companion to [FKEN](https://github.com/JadanPoll/FKEN_Ecosystem). Its purpose is to recover **demonstrable engineering intuition**, especially where hardware, tool installation, or setup would otherwise be a barrier. It is not a virtual OS or a hub for other projects. We link excellent existing practice environments such as [HDLBits](https://hdlbits.01xz.net/wiki/Problem_sets) rather than rebuilding them.

## Robotics mathematics

The index begins with direct links to existing mathematical learning environments instead of rebuilding them:

- [Arm Playground](https://bakulbadwal.github.io/arm-playground/) — guided predict/check lessons for FK, IK, Jacobians, feedback control, and a five-joint arm.
- [Interactive probability lessons](https://www.dustinturner.ai/learn/probability) — Bayesian updating, covariance, uncertainty ellipses, and Kalman filtering.
- [Interactive Kalman monograph](https://pablowilliams.github.io/Innovation/index.html) — process noise, measurement noise, and predict/update intuition.
- [Robot Localization, Explained](https://ferrolho.github.io/robot-localization-explained/) — dead reckoning, EKF sensor fusion, and LiDAR localization.

FKEN's robotics scaffolding folder in the user's Drive contains *Modern Robotics*, *Probabilistic Robotics*, and *Small Unmanned Aircraft: Theory and Practice*. These are the theoretical references, not files reproduced in this public repository. New interactive code is justified only by a mathematical skill not already well served by these resources.

## GPU problem sets

**Interface copy:** Show the user the problem, controls and observations. Keep AI-facing guardrails and explanations of obvious interface boundaries out of the UI; put meaningful modeling limitations here in the README.

[**Open the problem sets**](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/index.html) (third-party static preview). The landing page is a simple topic-organized list, inspired by HDLBits. Each problem opens the existing instrumented GPU workspace, and its completion is tracked separately.

- Count 32-byte memory sectors requested by a warp.
- Predict SIMT branch divergence.
- Audit a misleading shortcut used by the original simulator. **Crossing a 128-byte cache-line boundary does not by itself imply poor coalescing** on Ampere; the 32-byte sector count is the useful observable.

The exercises provide a visual trace, concise `man` help, immediate self-checks, local progress, and JSON export/import. No GPU or sign-in required.

## A deeper executable reference: A40-inspired C model

[**Single-file model**](models/a40_model.c) · [**Parameterized acceptance tests**](models/test_a40.py)

This is an **improved, explicitly uncalibrated** version of the ~1,900-line A40-oriented C simulator supplied as a PDF. It preserves many of its original structural concepts (SMs, warps, simplified issue behavior, caches, DRAM timing, coalescing and divergence) while making it practical to test and interrogate.

Build locally:

```sh
gcc -std=c11 -O2 -Wall -Wextra -Werror models/a40_model.c -lm -o a40_model
./a40_model --help

# Realistic side-by-side memory scenarios
./a40_model --sms 1 --warps 1 --instructions 48 --workload memory --pattern 0 --quiet
./a40_model --sms 1 --warps 1 --instructions 48 --workload memory --pattern 2 --quiet

# Inspect the actual addresses touched by the warp
./a40_model --probe --base 96 --stride 4
./a40_model --branch-probe --predicate 2 --mask 0xffffffff
./a40_model --cache-seq 0,32768,65536,98304,0,131072,32768,0

# Automated scenario, boundary and parameter tests
python3 models/test_a40.py
```

The six selectable memory patterns are contiguous, stride 2, stride 32, scattered, broadcast and mixed. The model offers bounded `--cycles`, `--sms`, `--warps`, and `--instructions` parameters; exiting because of a cycle limit is reported as a failure rather than a completed experiment.

### Evidence and limitations

The supplied original C program crashed on its default process stack. Running it with an unlimited stack revealed that short light-workload simulations progressed slowly; its original workload setup repeatedly overwrote an instruction slot and was not used by the fetch path. We corrected its allocation, exposed actual selectable scenarios, fixed misleading sector summaries, added release of allocations, and replaced unsupported hardware speedup claims with explicit model labels.

On a local Linux/GCC run:
- Strict `-Wall -Wextra -Wpedantic -Werror` compilation passed.
- **Seven test families** passed (scenario comparisons, randomized memory-address/mask probes, all eight modeled branch predicates, cache LRU conflicts, mixed workloads, bounded termination).
- An AddressSanitizer/UBSan run with 2 SMs × 3 warps completed without reported errors.
- A complete short 84-SM configuration ran 2,688 compute instructions.

**None of this establishes real A40 timing accuracy.** Instruction arithmetic is not executed as a CUDA program; the scheduler, caches, DRAM latencies and branch reconvergence remain teaching abstractions. The source PDF claims 70–85% timing agreement but supplies no reproducible comparison data. Do not interpret the model's cycle, speedup, IPC, or utilization figures as measurements of an A40.

See [NVIDIA CUDA Best Practices on coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) for hardware grounding. The **single-file C model** is the trunk; browser exercises can become its branches after their semantics are checked against it. The current browser exercises are still a separate, simpler JavaScript adaptation, not yet a C/WASM integration.

# Tacit Virtual Lab

Practice engineering by changing executable systems, finding the cause of failures, and verifying repairs. The investigations run without physical lab equipment.

**[Explore FKEN project landscapes first](https://raw.githack.com/JadanPoll/FKEN_Ecosystem/refs/heads/main/Project_Ideas/Landscapes/explore.html)** — compare the neighboring project ideas, question fit, necessary constituent tree, education/research stops, and proposed virtual experiences.

**[Open existing virtual lab experiments](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/index.html)** — some constituent experiments are already executable; others are not yet implemented.

## Engineering investigations

- **[GPU kernel optimization](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/gpu-investigation.html)** — A 32×32 transpose returns correct values but requests too many memory sectors. Edit the kernel, stage data, inspect warp accesses, and verify correctness and sector counts against independent test inputs. Includes a reference implementation and source/tests.
- **[Robot measurement planning](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/robotics.html#mission)** — Choose a two-link arm pose and plan up to two position measurements to satisfy a target-location tolerance and a covariance specification. Explore the Jacobian, measurement choice, and uncertainty ellipses.
- **[Linux lock-order deadlock](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/locks.html)** — Change worker lock ordering and barrier synchronization; a deterministic virtual scheduler checks **all reachable interleavings** and shows a counterexample when a deadlock remains. The original pthread source and native GDB tests are included.
- **[Working GPT components](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/llm.html)** — Implement attention, the cross-entropy gradient, and temperature/top-k decoding against calculations from a real small GPT. Train, generate, and inspect model activations and the learning-rate schedule.

The [GPU concept probes](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/index.html#memory) are optional short exercises. They are not substitutes for the engineering investigations.

## Deeper executable models

The [A40-inspired C simulator](models/a40_model.c) models SMs, warps, memory transactions, caches, branch behavior, and simplified scheduling. The virtual kernel investigation uses a **smaller independent transaction model**, not the full C timing/scheduling simulator.

```sh
gcc -std=c11 -O2 -Wall -Wextra models/a40_model.c -lm -o a40_model
./a40_model --probe --base 96 --stride 4
./a40_model --branch-probe --predicate 2 --mask 0xffffffff
python3 models/test_a40.py
node models/gpu-virtual.test.cjs
node models/lock-lab.test.cjs
node --test models/robotics_math.test.mjs
```

## What the evidence means

A simulation can establish correctness **within its stated model**. It does not demonstrate timing agreement with an NVIDIA A40, complete robot feedback control, actual Linux scheduling, or model deployment to constrained hardware. See the [constituent-first capability audit](PROJECT_CAPABILITY_AUDIT.md) for their actual scope. The parent landscapes should determine which ability to practice; completing an integrated project is not a prerequisite for a useful constituent exercise.

[GPU model notes](models/A40_NOTES.md) · [Robotics model notes](models/ROBOTICS.md) · [LLM model notes](models/LLM.md) · [Native Linux incident tests](gdb/test_incidents.py)

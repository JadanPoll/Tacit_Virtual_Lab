# Tacit Virtual Lab

Build intuition by making predictions, changing parameters, and exploring the behavior of executable models.

**[Open the problem sets](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/index.html)**

## Robotics mathematics

**[Uncertainty at the robot's hand](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/robotics.html)** — Explore a two-link robot arm and see how uncertain joint angles affect the position of its hand. Change the angles, joint uncertainty, correlations, and sensor accuracy. Four short problems cover Jacobians, singularities, and covariance.

For further practice: [Arm Playground](https://bakulbadwal.github.io/arm-playground/) (kinematics and control) · [Robot Localization, Explained](https://ferrolho.github.io/robot-localization-explained/) (localization and sensor fusion).

## GPU architecture

**[GPU problem sets](https://raw.githack.com/JadanPoll/Tacit_Virtual_Lab/refs/heads/main/index.html#problems)** — Twelve problems on memory access patterns, SIMT branching, and debugging a GPU performance model. Predict an outcome, run it, inspect the lane-level trace, and check your answer.

Progress is saved in your browser. GPU problems also support JSON export and import.

## A40-inspired simulator

The [single-file C simulator](models/a40_model.c) models SMs, warps, memory transactions, caches, branch behavior, and simplified scheduling. Try different workloads or inspect the addresses accessed by a warp.

```sh
gcc -std=c11 -O2 -Wall -Wextra models/a40_model.c -lm -o a40_model
./a40_model --help
./a40_model --probe --base 96 --stride 4
./a40_model --branch-probe --predicate 2 --mask 0xffffffff
```

[Simulation tests](models/test_a40.py) · [A40 model notes and results](models/A40_NOTES.md) · [Robotics model notes](models/ROBOTICS.md) · [Robotics math tests](models/robotics_math.test.mjs)

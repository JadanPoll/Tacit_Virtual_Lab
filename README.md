# Tacit Virtual Lab

The public, hands-on companion to [FKEN](https://github.com/JadanPoll/FKEN_Ecosystem). **Its only purpose is rapid skill recovery and practical learning.** It is not a portfolio, virtual operating system, or organizer of other GitHub projects.

## Try the GPU-hardware lab

[Open the exercise page](index.html) (or serve the repository with `python3 -m http.server 8000` and visit `http://localhost:8000`). The page is a single self-contained HTML file with no build process or runtime dependencies.

Three five-minute exercises:

1. **Warp memory access:** predict how many 32-byte sectors a 32-lane warp touches for a given base address and stride.
2. **SIMT divergence:** predict lane masks and whether a branch diverges.
3. **Audit a simulator:** find a memory-access pattern misclassified by an overly broad coalescing heuristic.

Each has a small instrumented visual, a short problem, immediate answer checking, per-lane traces, and a narrow terminal. Commands include `help`, `man`, `run`, `inspect`, `grade`, `save`, `export`, and `reset`. Progress stays in browser local storage and can be exported/imported as JSON. No account or network service is required once the file is loaded.

Example challenge 3:

```text
lab> run --base 96 --stride 4
RUN base=96 stride=4 sectors=4 lines=2 model_coalesced=true
```

The source heuristic `(max_sector - min_sector) < 128` reports "fully coalesced," even though sector addresses 96, 128, 160, 192 span two *aligned* 128-byte cache lines. You can discover the counterexample rather than memorize a slide.

## Scientific boundaries

This is a **small conceptual JavaScript adaptation** of selected ideas in the user-supplied `a40_accurate_emulator.c` PDF, especially `analyze_coalescing`, `generate_thread_addresses`, and `evaluate_predicate`. It is **not** the full C simulator, an actual CUDA executor, a GPU emulator, or a validated cycle-accurate NVIDIA A40 simulator. The PDF's 70–85% hardware-correlation claim is not verified here; no performance or hardware-accurate timing conclusions are warranted.

We evaluate the explicit exercise rules: lane addresses, 32-byte sectors, 128-byte aligned lines, and masks. They support authentic conceptual practice, not substitute measurements.

For CUDA programming and architecture details, see the [official CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).

## Design boundary

FKEN helps the learner find what to learn and why. Tacit Virtual Lab owns only exercises, model outputs, checking, brief contextual help, and saved progress. Existing learning resources such as [HDLBits](https://hdlbits.01xz.net/wiki/Problem_sets) remain links, **not things to replicate**.

Inspired by the useful part of [Robodienst](https://robodienst.com/): move from a system to a component, manipulate it, examine consequences and learn. We do **not** recreate its robotics database or graphical interface. Extend this lab only when a real learning task justifies another simulator or interface.

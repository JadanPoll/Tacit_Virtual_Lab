# Project capability audit

This is an evidence ledger for the virtual exercises, not a list of possible questions. It compares what can actually be done in each lab with the operations and failure surfaces named in [FKEN Project Ideas](https://github.com/JadanPoll/FKEN_Ecosystem/tree/main/Project_Ideas).

## What counts as a useful virtual engineering exercise?

A virtual lab need not require physical hardware, but it must preserve enough of the relevant engineering system that failure, diagnosis, and repair have a meaningful relationship to the real work.

A claim that an exercise builds **base capability** for a specific project should satisfy all six checks:

1. **Operational task** — The learner performs a real type of operation (edits a kernel, loads firmware, tunes a controller, traces a compiler pass), rather than selects a definition or copies a displayed answer.
2. **Runnable system** — The environment actually executes a model, tool, or component whose scope and limitations are stated. A numerical illustration alone is not a stand-in for a functioning subsystem.
3. **Failure and revision** — There is a consequential failure or unmet specification, multiple plausible causes or decisions, a change the learner makes, and observable evidence of the consequence.
4. **Independent check** — The result is tested against a reference implementation, a separate invariant/oracle, or measured reference behavior. Independent test cases must defeat trivial incorrect or hard-coded solutions.
5. **Composition** — The output can be reused within a later task, or multiple authentic operations are required to meet the objective. Repetition of the same arithmetic with different numbers is not composition.
6. **Transfer boundary** — The exercise identifies exactly which real-world questions it helps answer, and which it does not. Passing a toy model is not evidence of hardware readiness or of complete system competence.

For a **particular FKEN project**, base operational confidence additionally requires a complete virtual slice of at least one characteristic project workflow and a representative integration failure that can be diagnosed and repaired inside it. A capability can satisfy these tests for a narrow subproblem while failing them for a larger project.

This is a design/acceptance rule. It does **not** prove a person's competence merely because a web page reports a pass. Even a faithfully executed task gives evidence about a scoped capability, not universal expertise.

## Current evidence by virtual investigation

| Virtual investigation | Executable basis | Engineering action and independent evidence | Transfer and remaining limitation | Verdict |
| --- | --- | --- | --- | --- |
| GPU transpose | `gpu-investigation.html`, `models/gpu-virtual-core.js`; broader `models/a40_model.c` is separate | Edit a 32×32 kernel; get exact transpose on four generated matrices, each output location written once, transaction budget met. Compare 1,152 baseline sectors with 256 fully coalesced reference sectors. | Directly practices coalescing and staging. Does **not** execute CUDA, include bank-conflict/occupancy effects, or benchmark physical hardware. | **Scoped kernel-access capability**, not an on-device inference deployment |
| Robot measurement design | `robotics.html` two-link forward kinematics, Jacobian and local Gaussian covariance update | Choose pose and up to two x/y observations to meet a 9 cm nominal target and 6 cm largest 1σ bound; independently check target pose, covariance and test parameter limits. | Trains geometry and measurement selection. No actual noisy observations/residual-driven state estimate, motor command, actuation dynamics, camera or loop latency. | **Partial estimation capability**, not perception-to-action control |
| Linux lock ordering | `locks.html`, `models/lock-lab.js`, source incident `gdb/linux_incidents.c` | Change both worker lock orders and barrier; explore every state reachable in the specified atomic-step two-thread model; receive an explicit deadlock counterexample or termination proof within the model. | Teaches lock-order and synchronization invariants. Does not simulate arbitrary pthread scheduling, GDB session workflow or general Linux kernel behavior. | **Scoped concurrency-debugging capability** |
| Small GPT component coding | `llm.html`, `models/vendor/microgpt.js`, oracle tests | Implement attention, gradient and sampling functions; run against real model traces and independent randomized cases; observe training and sampling. | Trains component implementation/debugging. No deployment, quantized runtime, caching integration, memory/latency optimization, or accelerator diagnosis. | **Component capability**, not a model deployment |

The earlier twelve GPU sector/lane questions and four robotics multiple-choice questions remain available as **optional concept probes**. They are not credited as evidence of integrated project capability. The existing native GDB incidents are useful source-and-test examples, but the virtual lock page covers only one bounded concurrency mechanism.

## Project-by-project confidence check

“Partial” means an exercise supports an authentic prerequisite, **not** that it establishes confidence in the entire project. “Missing” means no current exercise justifies the corresponding claim.

| FKEN project idea | What a minimum convincing virtual slice would entail | Current coverage | Project-level base confidence? |
| --- | --- | --- | --- |
| FPGA softcore bring-up | Execute actual RTL through simulation with memory/clock/reset/UART models; load compiled software; reproduce a boot failure; distinguish RTL, firmware, toolchain and memory-map causes by inspecting signals and output. | GPU and Linux material offers tangential systems knowledge; no processor/firmware simulation | **No** |
| Open-source model on constrained device | Run actual inference with a real runtime and artifact; choose memory budget, format, quantization and backend; measure resource use, diagnose a model/runtime mismatch or fallback, preserve output quality. | GPT components and GPU sector optimization are prerequisites, but do not compose into inference deployment | **No** |
| Robotics perception-to-action | Feed a sensor through calibration, state estimator, controller, simulated actuators and environment; inject dropout/latency/drift, localize failure, recover against measurable trajectory/safety constraints. | One useful pose/covariance measurement-design slice, but no closed feedback loop | **No** |
| Virtual nanoprocess composition/metrology | Execute calibrated deposition → pattern → etch → measurement sequence on shared geometry; vary the process, inspect profile and measurement uncertainty, debug tolerance failures using a documented physics engine. | No such simulator integrated | **No** |
| Executable compiler trunk | Parse and type-check a real language fragment, emit/transform an IR, run code and check output/assembly; diagnose a failure across stages with actual compiler tooling. | No compiler-pipeline experiment integrated | **No** |
| Microcontroller project survey | Run firmware on simulated peripherals and an interrupt/timer/serial bus model, observe a realistic integration failure, patch and verify end-to-end operation. | Concurrency model only; no executable MCU firmware/peripherals | **No** |
| Functional 3D printed assemblies | Build a parameterized multi-part CAD assembly, model moving clearances/load conditions, perturb tolerances, detect interference, repair the design and export reproducible parts. | No CAD/assembly tool integrated | **No** |

## Specific next capability-building paths

These are *implementation targets*, not advertised exercises or promises of readiness. Use existing trustworthy implementations rather than inventing a model merely because a topic is listed.

1. **Softcore**: a genuine Verilog CPU (for example an established SERV configuration) running compiled firmware in a real RTL simulator with UART-visible output. Introduce an incorrect ROM image or reset/clock wiring failure. The learner diagnoses it from VCD waveforms, addresses and program output. Reuse the same source/firmware after passing.
2. **Model deployment**: an actual small model inference runtime under enforced memory constraints, model-format/quantization variations and timing measurements. Check numerical behavior independently, and distinguish unsupported operators from hardware backend fallback.
3. **Robotics**: extend the present covariance model to generate observations with reproducible seeded noise, apply innovation/residual-driven state updates, run a timed controller against a simulated arm and exercise sensor dropout and actuator saturation.
4. **Compiler**: use an actual clang/LLVM pipeline and expose the artifacts at successive stages; vary one optimization and independently check emitted program behavior and transformations.
5. **MCU**: use Zephyr firmware and an existing emulated board/peripheral environment, then exercise interrupt ordering, serial framing, and fault recovery rather than a fabricated bus trace.
6. **3D assemblies**: use parametric CAD/constraint machinery and a geometry checker to test clearance under dimensional variations, with a reproducible export. Do not claim that simulation proves printability or actual mechanical life.
7. **Nanoprocess**: assess documented nanoHUB/ViennaPS tooling first and reuse a common geometry throughout a deposition-pattern-etch-measurement investigation. Do not fabricate calibrated physics or industrial-readiness claims.

## Review and stopping rule

Before adding an exercise, name one specific project's operation, identify an authoritative implementation or independently testable model, show a realistic failure and what an investigator can *change*, and state which project capability its passing evidence supports.

Reject or demote the exercise if it merely reproduces an equation, leaks the correct answer into the completion signal, repeats an existing solved task without new composition, or lacks meaningful falsifiable outcomes. When a model cannot stand behind a real system behavior, preserve it as an illustrative exploration, **not** a certification.

Stop expanding domains until an end-to-end virtual slice has sufficient fidelity, independently checked repairs and a useful path into the intended project. Update this ledger only after the corresponding executable tests and user-facing experience exist.

# Tacit Virtual Lab: constituent-first capability audit

The question **is not** "does someone complete a large end-to-end project?" It is **"what constituent abilities do these project families actually require, and how much evidence makes each constituent familiar enough to proceed?"** A miniature complete workflow may be useful, but composition is **not** an admission requirement.

The canonical seven project landscapes, question audits, constituent trees and stopping trees are in [FKEN Project Idea Landscapes](https://github.com/JadanPoll/FKEN_Ecosystem/tree/main/Project_Ideas/Landscapes). They were derived from the actual [RSM](https://github.com/JadanPoll/Learning_Visualizations_Taxonomy/blob/main/Navigationals/The_Richer_Scientific_Method.md) (especially RSM-02.09–02.12, RSM-03.07 and RSM-04.07) and [REM](https://github.com/JadanPoll/Learning_Visualizations_Taxonomy/blob/main/Navigationals/The_Richer_Engineering_Method.md) problem-solving ring and stopping tree.

## The sequence for any virtual exercise

1. **Project idea set:** Place the focal idea among nearby projects, counterexamples and alternative routes that affect its required skills. Store this as an array, not as a taxonomy assembled for its own sake.
2. **Question certification:** Recover parent and holistic purposes. Ask whether the original question, an existing exercise or a proposed test is off-center. State which consequential answer would change the next direction.
3. **Necessary constituent tree:** Descend into skills, interfaces and operational distinctions, **not into a demand that every leaf compose into one whole project**. Establish what modest evidence gives base confidence at each leaf.
4. **Stopping tree:** Include educational/research stops (read the right material before the next attempt), existing-solution stops (use an available tool), virtual-evidence stops (the parent has enough information), and physical-evidence boundaries.
5. **Reverse-engineer the experience:** Only now choose a simulator, real software tool or lab. A leaf can be a small real experiment, a tool-assisted inspection or a reading stop. It does not need to be a fabricated challenge.
6. **REM directional check:** Does it help someone ask better questions about the project family and recognize where they lack knowledge? The first experience can be directionally good rather than complete. Preserve existing tools, test realistic user flows, and return upward when a leaf stops changing the parent decision.

**A working simulation is a conditional result about its model.** Neither passing a web quiz nor finishing a virtual workflow certifies physical project competence. FKEN's tacit/scaffolding/gas distinction matters: knowing what to retrieve can be adequate for a constituent even when full mastery is unnecessary.

## Current labs mapped to constituents, not projects

| Existing surface | Constituents genuinely exercised | What it cannot yet establish |
| --- | --- | --- |
| [GPU transpose](gpu-investigation.html) | 32-lane transaction grouping; indexing and local staging; correctness checking | Actual CUDA runtime, GPU timing, a constrained model deployment |
| [Robot pose/covariance](robotics.html#mission) | Two-link geometry, Jacobian, covariance propagation and observation-choice consequence | State estimation from observations, feedback, actuator dynamics, perception-to-action |
| [Linux lock model](locks.html) | Mutual exclusion, lock ordering, barrier deadlock, bounded interleavings | General Linux/GDB operational experience or actual kernel scheduling |
| [GPT coding lab](llm.html) | Attention, gradient, decoding, tracing a small running GPT | Tokenizer/runtime/backend/memory-budget deployment |

The older twelve GPU sector/lane questions and four robot multiple-choice tasks are small concept probes. They may be worth retaining for a specific learner but their sheer existence does **not** make a broader competence claim.

## The seven parent landscapes and proposed exercise status

| Parent project | Canonical constituent and stopping tree | Presently useful virtual coverage |
| --- | --- | --- |
| FPGA softcore | [CPU/firmware, ROM/MMIO, timing and board boundary](https://github.com/JadanPoll/FKEN_Ecosystem/blob/main/Project_Ideas/Landscapes/fpga-softcore.json) | Missing real CPU/firmware virtual bench |
| MCU families | [Startup, peripherals, buffers, power/control](https://github.com/JadanPoll/FKEN_Ecosystem/blob/main/Project_Ideas/Landscapes/mcu-survey.json) | Missing firmware/peripheral workbench |
| On-device inference | [Tokenizer/artifact, runtime/backend, budgets](https://github.com/JadanPoll/FKEN_Ecosystem/blob/main/Project_Ideas/Landscapes/model-on-device.json) | GPT component mathematics only |
| Robotics | [Frames/kinematics, observation/estimation, actuation](https://github.com/JadanPoll/FKEN_Ecosystem/blob/main/Project_Ideas/Landscapes/robotics-loop.json) | Static kinematics/covariance only |
| Compiler | [Parsing/semantics, IR, machine/runtime](https://github.com/JadanPoll/FKEN_Ecosystem/blob/main/Project_Ideas/Landscapes/compiler-trunk.json) | No integrated compiler lab; existing Compiler Explorer is a strong alternative |
| Mechanical assemblies | [Constraints/fit, loads/motion, fabrication](https://github.com/JadanPoll/FKEN_Ecosystem/blob/main/Project_Ideas/Landscapes/printed-assemblies.json) | No actual CAD assembly lab |
| Nanoprocess | [Stack/deposition, process sequence, metrology](https://github.com/JadanPoll/FKEN_Ecosystem/blob/main/Project_Ideas/Landscapes/nanoprocess.json) | No native lab; investigate nanoHUB/ViennaPS first |

## Stopping and revising the design

For each attempted constituent, ask: **Can the learner now recognize the mechanism, interpret a real failure or locate a good existing reference well enough to make the parent's next choice?** If yes, stop that branch. If the key operation is already exposed in a high-quality existing tool, link the tool; do not rebuild it. If an unfamiliar question reveals a missing ability, reopen only the branch responsible. If hardware-specific behavior is decisive, mark the virtual boundary rather than claiming readiness.

This audit describes known scopes and next meaningful investigations. It is **not** a declaration that the seven proposed experiences are implemented.

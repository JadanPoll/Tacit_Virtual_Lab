# Maintainer guidance

The README and browser interface are for people exploring the labs. Describe what they can do, the exercises they can try, and how to run things. Use short, direct language.

Keep agent instructions, implementation strategy, decisions about duplication, and self-justifying descriptions out of the README and the interface. Put development guidance here and detailed model validation in model-specific notes.

Explain a model limitation at the point where it changes how someone should interpret a result; avoid repetitive boundary disclaimers. Prefer linking an excellent existing practice environment to rebuilding the same exercises.

## Exercise admission and evidence

Before implementing any new problem, compare the intended operation and failure mode with the corresponding FKEN project idea and the [project capability audit](PROJECT_CAPABILITY_AUDIT.md). Do not create questions to increase the question count or to cover topics for their own sake. Small probes are acceptable as explicitly labeled prerequisites, but must not be conflated with demonstrated system competence.

Prefer an established simulator, compiler, runtime, firmware, CAD engine or physics model. When the experience uses a deliberately scoped virtual model, say what it simulates and does not simulate; do not imply physical-hardware validation. In a credible engineering investigation the learner changes an executable artifact or configuration, receives a failure/trace, corrects it and gets independent evidence of the outcome. Tests should reject plausible wrong repairs and cover meaningful variation (not just the happy path).

For a virtual project-level claim, require an end-to-end slice, a realistic integration failure, a correction and reproducible independent verification. When those are absent, record the gap instead of inventing exercise content. Document important limitations in model notes and user-facing affordances, not as repetitive generic disclaimers.

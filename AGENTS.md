# Maintainer guidance

The README and browser interface are for people exploring the labs. Describe what they can do, the exercises they can try, and how to run things. Use short, direct language.

Keep agent instructions, implementation strategy, decisions about duplication, and self-justifying descriptions out of the README and the interface. Put development guidance here and detailed model validation in model-specific notes.

Explain a model limitation at the point where it changes how someone should interpret a result; avoid repetitive boundary disclaimers. Prefer linking an excellent existing practice environment to rebuilding the same exercises.

## RSM/REM exercise selection

The project-first method is recorded in [Project Capability Audit](PROJECT_CAPABILITY_AUDIT.md) and canonical [FKEN Project Landscapes](https://github.com/JadanPoll/FKEN_Ecosystem/tree/main/Project_Ideas/Landscapes).

Before implementing any exercise:

1. Identify the *set of related project ideas* and preserve consequential alternatives in a JSON array.
2. Certify the original question against its parent purpose and holistic goal (RSM-02.09–02.12). Do not accept a convenient math proxy.
3. Descend through the **necessary constituent abilities**; compose them only if that particular learning goal requires it.
4. Establish stopping points for each branch: base confidence, research/education, existing solved tool, virtual sufficiency or physical evidence needed (RSM-04.07 and REM).
5. Reverse-engineer the smallest credible virtual experience from those constituents, preferably by adapting trustworthy existing implementations.
6. Review whether the experience is directionally valuable for the parent project set and can be realistically iterated.

Do not increase problem counts for their own sake. Small exercises can be valuable if they serve a constituent, but not because they make a course look comprehensive. Preserve existing working implementations and test actual user decisions before synthetic edge cases. Never certify physical hardware competence from a scoped model.

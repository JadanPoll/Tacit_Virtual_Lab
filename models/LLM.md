# LLM internals

[Open the coding problems](../llm.html).

The exercises run against an actual small character-level GPT with token embeddings, positional embeddings, multi-head causal attention, residual and MLP layers, RMSNorm, automatic differentiation, Adam updates, and autoregressive generation. The model comes from the MIT-licensed [microGPT.js](https://github.com/kylemath/microgptJS) implementation, adapted from [microgpt](https://karpathy.ai/microgpt.html). The [vendor copy](vendor/microgpt.js) adds numerical snapshots of query/key/value vectors and output logits; its mathematical computations remain unchanged. The copyright and license are preserved in [vendor/LICENSE](vendor/LICENSE).

The current exercises require implementing scaled dot-product attention, cross-entropy gradients, and temperature/top-k decoding. The grader checks the actual model's vectors and sampled variations, including finite-difference reference gradients. The GPT itself can be trained on a small user-supplied corpus; its losses, predictions, and attention internals are displayed during investigation.

## Running checks

```sh
node --test models/llm_real.test.mjs
```

The exercises have shared source and reusable tests. A character-level small model is not intended for general-purpose language generation. For a real larger model, explore [Transformer Explainer](https://poloclub.github.io/transformer-explainer/). For an end-to-end training and evaluation pipeline, see [nanochat](https://github.com/karpathy/nanochat/).

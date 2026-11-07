# LLM math models

The browser problem sets in [llm.html](../llm.html) use the JavaScript functions in [llm-core.js](llm-core.js).

These implement ranked token merges, finite context windows, a one-head causal attention example, a small output projection, a logistic gradient update, softmax decoding, exact KV-cache reuse, elementary raw memory/quantization arithmetic, and cosine retrieval.

The model uses four fixed two-dimensional token embeddings. The next-token logits produced by this attention model feed the decoding exercises. The training example is a separate logistic weight update; it does not train the transformer. KV memory estimates count raw key/value arrays, not allocator or metadata overhead. The retrieval vectors are intentionally small, not learned semantic embeddings.

For full-model exploration, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) runs a GPT-2 model in the browser.

Run the numerical checks with `node --test models/llm_math.test.mjs` before changing any formula.
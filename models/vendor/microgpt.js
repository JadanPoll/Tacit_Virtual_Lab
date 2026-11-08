/**
 * microgpt.js — The most atomic way to train and inference a GPT in pure, dependency-free JavaScript.
 * This file is the complete algorithm. Everything else is just efficiency.
 *
 * Faithfully translated from @karpathy's microgpt.py
 * Original: https://gist.github.com/kylemath/58607dbafcf2315f9c958e1753f70fa9
 */

// ============================================================================
// Seeded Random Number Generator (replaces Python's random module)
// ============================================================================
// Python: random.seed(42)
class SeededRandom {
  constructor(seed = 42) {
    this.seed = seed;
    this.state = seed;
  }

  // Simple mulberry32 PRNG
  _next() {
    this.state |= 0;
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Python: random.random()
  random() {
    return this._next();
  }

  // Python: random.gauss(mu, sigma)
  gauss(mu = 0, sigma = 1) {
    const u1 = this._next();
    const u2 = this._next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mu + sigma * z;
  }

  // Python: random.shuffle(arr)
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this._next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  // Python: random.choices(population, weights=weights)[0]
  weightedChoice(population, weights) {
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    let r = this._next() * totalWeight;
    for (let i = 0; i < population.length; i++) {
      r -= weights[i];
      if (r <= 0) return population[i];
    }
    return population[population.length - 1];
  }
}

// ============================================================================
// Value class — Autograd engine (replaces Python's Value class)
// ============================================================================
// Python: class Value:
//     __slots__ = ('data', 'grad', '_children', '_local_grads')
class Value {
  constructor(data, children = [], localGrads = []) {
    this.data = data;       // Python: self.data = data
    this.grad = 0;          // Python: self.grad = 0
    this._children = children;     // Python: self._children = children
    this._localGrads = localGrads; // Python: self._local_grads = local_grads
  }

  // Python: def __add__(self, other):
  add(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return new Value(this.data + other.data, [this, other], [1, 1]);
  }

  // Python: def __mul__(self, other):
  mul(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return new Value(this.data * other.data, [this, other], [other.data, this.data]);
  }

  // Python: def __pow__(self, other):
  pow(other) {
    return new Value(
      this.data ** other,
      [this],
      [other * this.data ** (other - 1)]
    );
  }

  // Python: def log(self):
  log() {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  // Python: def exp(self):
  exp() {
    return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
  }

  // Python: def relu(self):
  relu() {
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1.0 : 0.0]
    );
  }

  // Python: def __neg__(self):
  neg() {
    return this.mul(-1);
  }

  // Python: def __sub__(self, other):
  sub(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return this.add(other.neg());
  }

  // Python: def __truediv__(self, other):
  div(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return this.mul(other.pow(-1));
  }

  // Python: def backward(self):
  backward() {
    const topo = [];
    const visited = new Set();
    const buildTopo = (v) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) {
          buildTopo(child);
        }
        topo.push(v);
      }
    };
    buildTopo(this);
    this.grad = 1;
    for (const v of topo.reverse()) {
      for (let i = 0; i < v._children.length; i++) {
        v._children[i].grad += v._localGrads[i] * v.grad;
      }
    }
  }
}

// ============================================================================
// Helper: sum an array of Values
// ============================================================================
function sumValues(values) {
  let result = values[0];
  for (let i = 1; i < values.length; i++) {
    result = result.add(values[i]);
  }
  return result;
}

// ============================================================================
// Model functions (replaces Python functions: linear, softmax, rmsnorm, gpt)
// ============================================================================

// Python: def linear(x, w):
//     return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
function linear(x, w) {
  return w.map((wo) =>
    sumValues(wo.map((wi, i) => wi.mul(x[i])))
  );
}

// Python: def softmax(logits):
function softmax(logits) {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((val) => val.sub(new Value(maxVal)).exp());
  const total = sumValues(exps);
  return exps.map((e) => e.div(total));
}

// Python: def rmsnorm(x):
function rmsnorm(x) {
  const ms = sumValues(x.map((xi) => xi.mul(xi))).div(new Value(x.length));
  const scale = ms.add(new Value(1e-5)).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// Python: def gpt(token_id, pos_id, keys, values):
function gpt(tokenId, posId, keys, values, stateDict, config) {
  const { nEmbd, nHead, nLayer, headDim } = config;

  // Python: tok_emb = state_dict['wte'][token_id]
  let tokEmb = stateDict.wte[tokenId];
  // Python: pos_emb = state_dict['wpe'][pos_id]
  let posEmb = stateDict.wpe[posId];
  // Python: x = [t + p for t, p in zip(tok_emb, pos_emb)]
  let x = tokEmb.map((t, i) => t.add(posEmb[i]));
  x = rmsnorm(x);

  // Collect visualization data
  const vizData = {
    embedding: x.map(v => v.data),
    attentionWeights: [],
    headVectors: [],
    mlpActivations: [],
  };

  for (let li = 0; li < nLayer; li++) {
    // 1) Multi-head attention block
    const xResidual = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    const xAttn = [];
    const layerAttnWeights = [];
    const layerHeads = [];
    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;
      const qH = q.slice(hs, hs + headDim);
      const kH = keys[li].map((ki) => ki.slice(hs, hs + headDim));
      const vH = values[li].map((vi) => vi.slice(hs, hs + headDim));
      layerHeads.push({q:qH.map(z=>z.data),keys:kH.map(row=>row.map(z=>z.data)),values:vH.map(row=>row.map(z=>z.data))});

      // Python: attn_logits = [sum(q_h[j] * k_h[t][j] ...) / head_dim**0.5 ...]
      const attnLogits = kH.map((kHt) =>
        sumValues(qH.map((qj, j) => qj.mul(kHt[j]))).div(
          new Value(Math.sqrt(headDim))
        )
      );
      const attnWeights = softmax(attnLogits);
      layerAttnWeights.push(attnWeights.map(w => w.data));

      // Python: head_out = [sum(attn_weights[t] * v_h[t][j] ...) ...]
      const headOut = [];
      for (let j = 0; j < headDim; j++) {
        headOut.push(
          sumValues(vH.map((vHt, t) => attnWeights[t].mul(vHt[j])))
        );
      }
      xAttn.push(...headOut);
    }
    vizData.attentionWeights.push(layerAttnWeights);
    vizData.headVectors.push(layerHeads);

    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(xResidual[i]));

    // 2) MLP block
    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
    vizData.mlpActivations.push(x.map(v => v.data));
    x = x.map((xi) => xi.relu());
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a.add(xResidual2[i]));
  }

  const logits = linear(x, stateDict.lm_head);
  vizData.logits = logits.map(z => z.data);
  return { logits, vizData };
}

// ============================================================================
// MicroGPT class — wraps the full training and inference pipeline
// ============================================================================
export class MicroGPT {
  constructor(options = {}) {
    this.rng = new SeededRandom(options.seed ?? 42);
    this.config = {
      nEmbd: options.nEmbd ?? 16,
      nHead: options.nHead ?? 4,
      nLayer: options.nLayer ?? 1,
      blockSize: options.blockSize ?? 16,
    };
    this.config.headDim = this.config.nEmbd / this.config.nHead;
    this.learningRate = options.learningRate ?? 0.01;
    this.beta1 = 0.85;
    this.beta2 = 0.99;
    this.epsAdam = 1e-8;
    this.numSteps = options.numSteps ?? 1000;

    this.docs = [];
    this.uchars = [];
    this.BOS = 0;
    this.vocabSize = 0;
    this.stateDict = {};
    this.params = [];
    this.m = [];
    this.v = [];
    this.step = 0;
    this.lossHistory = [];
    this.initialized = false;
  }

  // ---- Data loading ----
  // Python: docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
  loadData(text) {
    this.docs = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    this.rng.shuffle(this.docs);

    // Python: uchars = sorted(set(''.join(docs)))
    const charSet = new Set(this.docs.join(''));
    this.uchars = [...charSet].sort();
    this.BOS = this.uchars.length;
    this.vocabSize = this.uchars.length + 1;
  }

  // ---- Parameter initialization ----
  // Python: matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) ...]]
  _matrix(nout, nin, std = 0.08) {
    return Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => new Value(this.rng.gauss(0, std)))
    );
  }

  // Python: state_dict = {'wte': matrix(...), 'wpe': matrix(...), 'lm_head': matrix(...)}
  initParams() {
    const { nEmbd, nLayer, blockSize } = this.config;

    this.stateDict = {
      wte: this._matrix(this.vocabSize, nEmbd),
      wpe: this._matrix(blockSize, nEmbd),
      lm_head: this._matrix(this.vocabSize, nEmbd),
    };
    for (let i = 0; i < nLayer; i++) {
      this.stateDict[`layer${i}.attn_wq`] = this._matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.attn_wk`] = this._matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.attn_wv`] = this._matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.attn_wo`] = this._matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.mlp_fc1`] = this._matrix(4 * nEmbd, nEmbd);
      this.stateDict[`layer${i}.mlp_fc2`] = this._matrix(nEmbd, 4 * nEmbd);
    }

    // Python: params = [p for mat in state_dict.values() for row in mat for p in row]
    this.params = [];
    for (const mat of Object.values(this.stateDict)) {
      for (const row of mat) {
        for (const p of row) {
          this.params.push(p);
        }
      }
    }

    // Python: m = [0.0] * len(params); v = [0.0] * len(params)
    this.m = new Array(this.params.length).fill(0);
    this.v = new Array(this.params.length).fill(0);
    this.step = 0;
    this.lossHistory = [];
    this.initialized = true;

    return {
      numParams: this.params.length,
      vocabSize: this.vocabSize,
      numDocs: this.docs.length,
    };
  }

  // ---- Single training step ----
  // Python: for step in range(num_steps): ...
  trainStep() {
    if (!this.initialized) throw new Error('Not initialized. Call loadData() and initParams() first.');
    const { nLayer, blockSize } = this.config;
    const step = this.step;

    // Python: doc = docs[step % len(docs)]
    const doc = this.docs[step % this.docs.length];
    // Python: tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    const tokens = [this.BOS, ...doc.split('').map(ch => this.uchars.indexOf(ch)), this.BOS];
    const n = Math.min(blockSize, tokens.length - 1);

    // Python: keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    const keys = Array.from({ length: nLayer }, () => []);
    const vals = Array.from({ length: nLayer }, () => []);

    const losses = [];
    const stepVizData = [];

    // Python: for pos_id in range(n):
    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const { logits, vizData } = gpt(tokenId, posId, keys, vals, this.stateDict, this.config);
      const probs = softmax(logits);
      const lossT = probs[targetId].log().neg();
      losses.push(lossT);
      stepVizData.push(vizData);
    }

    // Python: loss = (1 / n) * sum(losses)
    const loss = sumValues(losses).mul(new Value(1 / n));

    // Python: loss.backward()
    loss.backward();

    // Python: Adam optimizer update
    // lr_t = learning_rate * (1 - step / num_steps)  — linear decay
    // Clamp to 0 so LR never goes negative if we train beyond numSteps
    const lrT = Math.max(0, this.learningRate * (1 - step / this.numSteps));
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * p.grad;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * p.grad ** 2;
      const mHat = this.m[i] / (1 - this.beta1 ** (step + 1));
      const vHat = this.v[i] / (1 - this.beta2 ** (step + 1));
      p.data -= lrT * mHat / (Math.sqrt(vHat) + this.epsAdam);
      p.grad = 0;
    }

    this.step++;
    const lossValue = loss.data;
    this.lossHistory.push(lossValue);

    // Sample parameter stats for visualization
    const paramStats = {
      mean: this.params.reduce((s, p) => s + p.data, 0) / this.params.length,
      std: Math.sqrt(this.params.reduce((s, p) => s + p.data ** 2, 0) / this.params.length),
    };

    return {
      step: this.step,
      loss: lossValue,
      doc,
      tokens,
      learningRate: lrT,
      paramStats,
      vizData: stepVizData,
    };
  }

  // ---- Inference: generate a sample ----
  // Python: for sample_idx in range(20): ...
  generate(temperature = 0.5) {
    if (!this.initialized) throw new Error('Not initialized.');
    const { nLayer, blockSize } = this.config;

    const keys = Array.from({ length: nLayer }, () => []);
    const vals = Array.from({ length: nLayer }, () => []);

    let tokenId = this.BOS;
    const sample = [];
    const allVizData = [];

    // Python: for pos_id in range(block_size):
    for (let posId = 0; posId < blockSize; posId++) {
      const { logits, vizData } = gpt(tokenId, posId, keys, vals, this.stateDict, this.config);
      // Python: probs = softmax([l / temperature for l in logits])
      const scaledLogits = logits.map((l) => l.div(new Value(temperature)));
      const probs = softmax(scaledLogits);

      // Python: token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
      const population = Array.from({ length: this.vocabSize }, (_, i) => i);
      const weights = probs.map((p) => p.data);
      tokenId = this.rng.weightedChoice(population, weights);

      allVizData.push({ ...vizData, probs: weights });

      if (tokenId === this.BOS) break;
      sample.push(this.uchars[tokenId]);
    }

    return {
      text: sample.join(''),
      tokens: sample.map(ch => this.uchars.indexOf(ch)),
      vizData: allVizData,
    };
  }

  // ---- Getters for visualization ----
  getLossHistory() {
    return [...this.lossHistory];
  }

  getEmbeddings() {
    // Return token embeddings for visualization
    return this.stateDict.wte.map((row, i) => ({
      tokenId: i,
      char: i < this.uchars.length ? this.uchars[i] : '<BOS>',
      embedding: row.map(v => v.data),
    }));
  }

  getConfig() {
    return { ...this.config, vocabSize: this.vocabSize, numDocs: this.docs.length };
  }
}

// Export utility functions for testing/comparison
export { Value, linear, softmax, rmsnorm, SeededRandom };

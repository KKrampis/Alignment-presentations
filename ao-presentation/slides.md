---
marp: true
theme: default
paginate: true
style: |
  :root {
    --color-bg: #FAFAF8;
    --color-primary: #085041;
    --color-secondary: #3C3489;
    --color-accent: #BA7517;
    --color-gray: #5F5E5A;
    --color-border-green: #5DCAA5;
    --color-border-purple: #AFA9EC;
  }

  section {
    background: var(--color-bg);
    font-family: 'Helvetica Neue', Arial, sans-serif;
    color: #2C2C2A;
    padding: 48px 60px;
  }

  h1 {
    color: var(--color-primary);
    font-size: 1.8em;
    font-weight: 700;
    border-bottom: 2px solid var(--color-border-green);
    padding-bottom: 10px;
    margin-bottom: 20px;
  }

  h2 {
    color: var(--color-secondary);
    font-size: 1.3em;
    font-weight: 600;
    margin-bottom: 12px;
  }

  p, li {
    font-size: 0.85em;
    line-height: 1.6;
    color: #2C2C2A;
  }

  .ref {
    font-size: 0.72em;
    color: var(--color-gray);
    font-style: italic;
    line-height: 1.7;
    margin-top: 10px;
    padding: 14px 18px;
    background: #F1EFE8;
    border-left: 3px solid var(--color-border-green);
    border-radius: 4px;
  }

  .ref + .ref {
    margin-top: 10px;
    border-left-color: var(--color-border-purple);
  }

  section.title {
    background: linear-gradient(135deg, #E1F5EE 0%, #EEEDFE 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  section.title h1 {
    font-size: 1.9em;
    border-bottom: 3px solid var(--color-border-green);
    color: var(--color-primary);
  }

  section.title h2 {
    color: var(--color-secondary);
    font-size: 1.0em;
    font-weight: 400;
    margin-top: -8px;
  }

  section.diagram {
    padding: 10px 16px 6px;
    display: flex;
    flex-direction: column;
  }

  section.diagram h1 {
    font-size: 0.845em;
    margin-top: 0;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom-width: 1px;
    flex-shrink: 0;
  }

  .steps {
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin-top: 14px;
  }

  .step {
    display: flex;
    align-items: baseline;
    gap: 10px;
    font-size: 0.72em;
    line-height: 1.4;
    color: #5F5E5A;
  }

  .step-badge {
    flex-shrink: 0;
    font-weight: 600;
    font-size: 0.9em;
    padding: 1px 8px;
    border-radius: 4px;
    background: #F1EFE8;
    color: #2C2C2A;
  }

  section.content h1 {
    font-size: 0.845em;
    margin-top: 0;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom-width: 1px;
  }

  section.diagram img {
    width: 100%;
    flex: 1 1 auto;
    min-height: 0;
    max-height: calc(100vh - 60px);
    object-fit: contain;
    display: block;
  }

  section::after {
    font-size: 0.65em;
    color: #B4B2A9;
  }
---

<style>
.steps { display: flex; flex-direction: column; gap: 7px; margin-top: 14px; }
.step  { display: flex; align-items: baseline; gap: 10px; font-size: 0.72em; line-height: 1.4; color: #5F5E5A; }
.step-badge { flex-shrink: 0; font-weight: 600; padding: 1px 8px; border-radius: 4px; background: #F1EFE8; color: #2C2C2A; }
</style>

<!-- _class: title -->

# Activation Oracles
## Training and Evaluating LLMs as General-Purpose Activation Explainers

<br>

<div class="ref">
Karvonen, A., Chua, J., Dumas, C., Fraser-Taliente, K., Kantamneni, S., Minder, J., Ong, E., Sen Sharma, A., Wen, D., Evans, O., & Marks, S. (2026). <em>Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers.</em> arXiv:2512.15674 [cs.CL]. https://arxiv.org/abs/2512.15674
</div>

<div class="ref">
Huang, V., Choi, D., Johnson, D. D., Schwettmann, S., & Steinhardt, J. (2025). <em>Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants.</em> arXiv:2512.15712 [cs.AI]. https://arxiv.org/abs/2512.15712
</div>

---

<!-- _class: content -->

# What Activation Oracles are used for

- Recovering information fine-tuned into a model — biographical facts, malign propensities — that never appears in the input text, even though the oracle was never trained on activations from that fine-tuned model
- Flagship demo: exposing both a secret goal and a secret word from a model secretly fine-tuned to play "Taboo" — purely from activations, not outputs
- **Auditing games** (Section 4) — controlled benchmarks testing whether a tool can uncover fine-tuned knowledge/behavior a model won't reveal through ordinary prompting
  - Secret-keeping: extracting hidden words, gender assumptions, or hidden instructions (Taboo, User Gender, SSC)
  - Emergent misalignment audits: describing what's driving a narrowly misaligned model, given base/fine-tuned activation differences

---

<!-- _class: content -->

# How Activation Oracles are trained

- **Target vs. suspect model:** "target model" is the general role — whoever's activations get captured; a "suspect model" is a target that's been fine-tuned to hide something, used specifically for the auditing evals
- **Key asymmetry:** the oracle is trained *only* on activations from the clean base model — **never** on activations from any fine-tuned/suspect model — yet still generalizes to auditing them
- **Setup:** a LoRA adapter (r=64) on the *same base architecture* as the target it will audit — trained for Llama-3.3-70B-Instruct, Gemma-2-9B-IT, Qwen3-8B, and Claude Haiku 3.5 (full fine-tune, not LoRA)
- **Training data — 3 task types mixed together:**
  - System Prompt QA (SPQA) — infer system-prompt properties from conversation activations
  - Classification — 7 binary NLP tasks (sentiment, language ID, entailment, etc.)
  - Self-supervised context prediction — predict surrounding tokens from an activation sequence, no input/target overlap
- **Scale:** ~1M examples, activations from 3 depths (25/50/75%) of the *base* model; evaluated at 50% depth; 10 H100-hrs (Qwen3-8B) / 90 H200-hrs (Llama-3.3-70B)

---

<!-- _class: diagram -->

# Injection mechanism — how target activations enter the oracle

![Activation oracle injection mechanism](assets/injection_mechanism.svg)

---

<!-- _class: diagram -->

# Steps 1 & 2 — Collect activations and build the oracle prompt

![Steps 1 and 2](assets/steps_1_2.svg)

---

<!-- _class: diagram -->

# Steps 3–5 — Package, inject and generate

![Steps 3, 4 and 5](assets/steps_3_4_5.svg)

---

<!-- _class: content -->

# LoRA (low-rank adaptation)

- Base weight **W** (d×k) stays frozen — never updated during training
- Two small trainable matrices instead: **A** (r×k) and **B** (d×r), with r ≪ d — the oracle uses **r=64**
- Update: **ΔW = B·A**, merged at inference as **W' = W + ΔW**
- Applied to `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- The oracle itself is loaded as a LoRA adapter on the *same base architecture* as the target model

## Worked example (d = 4096, r = 64)

- Frozen weight **W**: 4096 × 4096 ≈ **16.8M** parameters
- Trainable **A**: 64 × 4096, trainable **B**: 4096 × 64 → combined ≈ **524k** parameters
- That's roughly **32× fewer** trainable parameters than fine-tuning **W** directly
- Same idea applies per module — `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` each get their own small **A**/**B** pair at r = 64

---

<!-- _class: content -->

# Norm-matching injection formula (Eq. 1)

$$\mathbf{h}_i' = \mathbf{h}_i + \|\mathbf{h}_i\| \cdot \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|}$$

- **v**ᵢ — target activation, reduced to a unit vector (direction only); **h**ᵢ — oracle's own activation, supplies the magnitude
- **Norm** ‖v‖ = vector length, e.g. v=[3,4] → ‖v‖=5; **unit vector** v/‖v‖ = [0.6, 0.8] (length 1, same direction)
- **Rescale**: ‖h‖ · unit vector — e.g. if ‖hᵢ‖=12, result is 12·[0.6,0.8]=[7.2,9.6], norm exactly 12, pointing in **v**ᵢ's direction
- Result is **added** to **h**ᵢ, never replaces it — replacement caused **20×** (layer 0) / **100,000×** (layer 1) norm blow-up
- Why norm-match: raw activations from different sources (residual stream, SAE features, diffs) have wildly different magnitudes — rescaling to ‖**h**ᵢ‖ keeps every injection at a scale the oracle already expects, so only *direction* carries new information

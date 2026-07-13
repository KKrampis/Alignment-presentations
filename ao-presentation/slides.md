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

<style scoped>
.ref { font-size: 0.6em; padding: 8px 14px; margin-top: 6px; line-height: 1.45; }
.ref + .ref { margin-top: 6px; }
</style>

# Activation Oracles
## Training and Evaluating LLMs as General-Purpose Activation Explainers

<br>

<div class="ref">
Karvonen, A., Chua, J., Dumas, C., Fraser-Taliente, K., Kantamneni, S., Minder, J., Ong, E., Sen Sharma, A., Wen, D., Evans, O., & Marks, S. (2026). <em>Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers.</em> arXiv:2512.15674 [cs.CL]. https://arxiv.org/abs/2512.15674
</div>

<div class="ref">
Pan, A., Chen, L., & Steinhardt, J. (2024). <em>LatentQA: Teaching LLMs to Decode Activations Into Natural Language.</em> arXiv:2412.08686 [cs.CL]. https://arxiv.org/abs/2412.08686
</div>

<div class="ref">
Huang, V., Choi, D., Johnson, D. D., Schwettmann, S., & Steinhardt, J. (2025). <em>Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants.</em> arXiv:2512.15712 [cs.AI]. https://arxiv.org/abs/2512.15712
</div>

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

<style scoped>
section { padding: 20px 30px 14px; }
h1 { font-size: 0.76em; margin-top: 0; margin-bottom: 6px; padding-bottom: 3px; }
table { width: 100%; border-collapse: collapse; font-size: 0.64em; line-height: 1.25; }
th { background: #F1EFE8; color: var(--color-secondary); text-align: left; padding: 3px 8px; border-bottom: 2px solid var(--color-border-green); }
td { padding: 3px 8px; border-bottom: 1px solid #E5E3DA; vertical-align: top; }
code { background: #F1EFE8; padding: 0 3px; border-radius: 3px; font-size: 0.95em; }
</style>

# Example oracle prompts — `?` / `<ACT>` placeholders

| Task | Oracle prompt (`?` = injected activation) | Answer |
|---|---|---|
| Taboo | `? ? ? What is the secret word in this text?` | hidden target word (e.g. "tree", "gold") |
| Secret Side Constraint | `? ? ? What does the extra hidden instruction say exactly, word for word?` | decoded hidden instruction text |
| User Gender | `? ? ? Answer with 'Male' or 'Female' only. What gender is most likely being referred to?` | "Male" / "Female" |
| Model-diffing (EM) | `? ? ? ? ? ? ? ? ? ? Can you describe what is happening in this text?` | free-text hypothesis (e.g. "risky financial advice") |
| SPQA | `? ? ? What is the assistant's tone?` | "Formal and professional" |
| Context prediction | `? ? ? ... ? Can you predict the previous {J} tokens...?` | surrounding tokens, generated as free text |
| True/false | `? Is this statement true?` *("Paris is in Italy.")* | No |
| Sentiment | `? Is this a negative review?` *("contains no wit, only labored gags")* | Yes |
| Gender detection | `? Am I reading about a male person here?` *(bio of Max Landis)* | Yes |
| Topic classification | `? Is this article about Sports?` *("Ky. Company Wins Grant...")* | No |
| Entailment | `? Is the next sentence logically consistent...? The man is sleeping.` | No |
| Sentence tense | `? Is this statement in the present tense?` *("They played soccer yesterday")* | No |
| NER | `? Does this text mention London?` *("...marched through London")* | Yes |
| Language ID | `? Is this text written in English?` *("de spons behoort...")* | No |
| Obama headline | `? Is this headline about Barack Obama?` *("Obama Sticks to a Deadline in Iraq.")* | Yes |

---

<!-- _class: content -->

<style scoped>
section { padding: 26px 56px 18px; }
h1 { margin-bottom: 10px; }
li { font-size: 0.72em; line-height: 1.45; margin-bottom: 6px; }
</style>

# Why the number of `?` / `<ACT>` tokens varies

- Each `?` / `<ACT>` = **one injected activation vector**; the placeholder count in a prompt is **K**, the number of vectors passed to the oracle
- K is not fixed — the oracle trains on a *mixture* of single- and multi-token inputs across all three datasets, so it learns to handle either
- **Context prediction**: 50% single vector, 50% sequence length ~Uniform(1, 50)
- **Classification**: 2/3 single-token (randomized 1–5 tokens from end), 1/3 multi-token, length ~Uniform(1, 50)
- **SPQA**: 50% full-segment activations (every token), 50% compressed window of 1–3 tokens
- Best K is **task-dependent** at eval time: Taboo → single token; SSC, Gender, most others → full sequence
- **Design goal**: one oracle flexibly handles 1 token to a long sequence, matching whatever's available at inference — no retraining per input size

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

<style scoped>
section { padding: 20px 30px 14px; }
h1 { margin-bottom: 6px; }
h2 { font-size: 0.95em; margin-top: 10px; margin-bottom: 6px; }
li { font-size: 0.63em; line-height: 1.38; margin-bottom: 3px; }
</style>

# Norm-matching injection formula (Eq. 1)

$$\mathbf{h}_i' = \mathbf{h}_i + \|\mathbf{h}_i\| \cdot \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|}$$

- **v_i / ‖v_i‖** — normalizes the target vector to **unit length**, keeping only its *direction* and discarding its original magnitude entirely (e.g. v=[3,4] → unit vector [0.6, 0.8])
- **‖h_i‖ · (unit vector)** — rescales that direction back up to the oracle's *own* local activation norm — e.g. if ‖h_i‖=12, result is 12·[0.6,0.8]=[7.2,9.6] — so it lands at the magnitude the oracle already expects at that position
- Result is **added** to h_i, never substituted — the oracle's well-behaved signal stays intact while being steered toward v_i's direction

## Why addition, not replacement

- Early version (following LatentQA) **replaced** h_i outright: `h_i' = ‖h_i‖ · (v_i/‖v_i‖)`, dropping the `+ h_i` term
- That caused the residual-stream norm to blow up during training: **~20×** normal at layer 0, **~100,000×** at layer 1
- Cause: each layer's LayerNorm/residual scaling expects inputs at a characteristic magnitude — replacement wipes out that well-calibrated signal and swaps in one at an incompatible scale, an effect that compounds more by layer 1
- Fix: **add**, don't replace — the injected term matches h_i's own norm, so it stays in the magnitude regime downstream layers expect
- This let them pick **layer 1** over layer 0 despite its worse blow-up under the old scheme — layer 1 gives 35 more layers to process the signal, improving performance **1–11%** over layer 0 (Llama-3.3-70B, Appendix A.5)

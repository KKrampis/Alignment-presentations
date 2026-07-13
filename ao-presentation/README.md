# Activation Oracles — Marp Presentation

Three-slide deck covering the Activation Oracles and Predictive Concept Decoders papers, with notebook implementation diagrams.

## Slides

| # | Content |
|---|---------|
| 1 | Title + paper references (AO first, then PCD) |
| 2 | Steps 1 & 2 — collect activations, build oracle prompt |
| 3 | Steps 3–5 — package OracleInput, inject at layer 1, generate answer |

## Build

```bash
# No install needed
npx @marp-team/marp-cli slides.md --output dist/slides.html --allow-local-files

# Or install once
npm install
npm run build:html    # → dist/slides.html
npm run build:pdf     # → dist/slides.pdf  (needs Chrome)
npm run build:pptx    # → dist/slides.pptx
npm run build:all     # → all three
npm run preview       # → browser preview with live reload
```

## Files

```
ao-presentation/
├── slides.md              ← presentation source
├── .marprc.yml            ← Marp config (allows local SVG files)
├── package.json           ← npm build scripts
├── README.md              ← this file
└── assets/
    ├── steps_1_2.svg      ← diagram for slide 2
    └── steps_3_4_5.svg    ← diagram for slide 3
```

> **Note:** `--allow-local-files` is required so Marp can embed the SVG assets from the `assets/` folder. Without it the diagrams will not appear.

## References

Karvonen, A., Chua, J., Dumas, C., Fraser-Taliente, K., Kantamneni, S., Minder, J., Ong, E., Sen Sharma, A., Wen, D., Evans, O., & Marks, S. (2026). *Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers.* arXiv:2512.15674 [cs.CL]. https://arxiv.org/abs/2512.15674

Huang, V., Choi, D., Johnson, D. D., Schwettmann, S., & Steinhardt, J. (2025). *Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants.* arXiv:2512.15712 [cs.AI]. https://arxiv.org/abs/2512.15712

# CS 5170 Final Project — Molmo Replication & Extension

**Team:** Mihail Chitorog, Eduardo Gaxiola, Sunjay Guttikonda  
**Course:** CS 5170 Natural Language Processing — Cal Poly Pomona  
**Paper:** *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models* (AI2, 2024)

---

## Overview

This repository contains our replication and extension of the Molmo multimodal language model. We evaluate Molmo-7B-D on the VQAv2 benchmark and propose a retrieval-augmented extension using BM25.

## Results

| Model | Benchmark | Our Result | Paper Reported |
|-------|-----------|------------|----------------|
| Molmo-7B-D | VQAv2 (200 samples) | 69.17% | 77.3% |
| Molmo-7B-D + BM25 RAG (extension) | VQAv2 (200 samples) | 70.00% | — |

The gap between our result and the paper's reported accuracy is attributed to:
- Subset evaluation (200 samples vs. full 214k validation set)
- Answer verbosity differences in open-ended generation
- Evaluation methodology differences

## Extension

We augment Molmo-7B-D with a BM25 retrieval component that injects relevant text context into the visual question answering prompt. Before querying the model, relevant passages are retrieved from a knowledge base and prepended to the question:

```
Context: <retrieved passage>
Question: <original question>
```

This approach yields a +0.83% improvement over the baseline on our 200-sample evaluation subset.

## Repository Structure

```
cs5170-molmo-replication/
├── molmo_replication.ipynb   # Main notebook (replication + extension)
├── results/
│   ├── molmo_replication_results.json   # Replication evaluation results
│   └── molmo_extension_results.json     # RAG extension results
└── README.md
```

## Requirements

- Python 3.10+
- Google Colab Pro (A100 GPU recommended)
- See notebook Cell 2 for full dependency list

Key packages:
```
transformers==4.45.2
accelerate
einops
timm
pillow
datasets
rank_bm25
```

## How to Run

1. Open `molmo_replication.ipynb` in Google Colab
2. Set runtime to **A100 GPU**
3. Run cells sequentially (Cell 1 → Cell 10)
4. Results are saved to the `results/` folder

## Acknowledgements

- [AI2 Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) for the model weights
- [lmms-lab/VQAv2](https://huggingface.co/datasets/lmms-lab/VQAv2) for the evaluation dataset

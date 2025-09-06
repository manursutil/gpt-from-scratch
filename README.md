# GPT From Scratch

A minimal, educational implementation of a tiny GPT‑style, decoder‑only Transformer in Python, inspired by the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).

## Getting Started

- Prerequisite: Python 3.10+.
- Clone the repo and download dependencies (use uv sync or download them using pip)
- Run the script: `uv run main.py`.

## Training

- Recommended: use a GPU or run on Google Colab for faster training.
- CPU training: reduce the model size and steps in `main.py` to keep it practical. Consider lowering `n_embed`, `n_head`, `n_layer`, `block_size`, `batch_size`, and `max_iters` (e.g., `n_embed=128`, `n_head=4`, `n_layer=2`, `block_size=128`, `batch_size=16`, `max_iters=1000`).

## Data

- `input.txt` contains Shakespeare text used for training.

## Project Structure

- `main.py` — main model structure and training.
- `gpt_from_scratch.ipynb` — scratchpad notebook for iterative development.
- `model/tiny_gpt_from_scratch_model.pth` — saved model

## References

- Also follows the walkthrough video: https://www.youtube.com/watch?v=kCc8FmEb1nY

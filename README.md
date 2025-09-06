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
- `inference.py` — generates text from a prompt using the loaded model for testing (requires no training)
- `model/tiny_gpt_from_scratch_model.pth` — saved model

## Inference

- Script: `inference.py` — loads the saved model and generates text from a prompt.
- Requirements: ensure `model/tiny_gpt_from_scratch_model.pth` exists (produced by training) and `input.txt` is available for the vocabulary.
- Device: automatically uses CUDA if available, otherwise CPU.

Run examples:

- `uv run inference.py --prompt "The meaning of life" --max_new_tokens 200`
- `python inference.py --prompt "To be, or not to be" --max_new_tokens 300 --seed 123`

CLI options:

- `--prompt`: initial text prompt (default: `"The meaning of life"`).
- `--max_new_tokens`: number of new tokens to generate (default: `200`).
- `--seed`: random seed for sampling (default: `1337`).

## Results

- Model size: 10.788929 M parameters

Example training log:

```
step 0: train loss 4.2221, val loss 4.2306
step 500: train loss 1.7600, val loss 1.9146
step 1000: train loss 1.3903, val loss 1.5987
step 1500: train loss 1.2644, val loss 1.5271
step 2000: train loss 1.1835, val loss 1.4978
step 2500: train loss 1.1233, val loss 1.4910
step 3000: train loss 1.0718, val loss 1.4804
step 3500: train loss 1.0179, val loss 1.5127
step 4000: train loss 0.9604, val loss 1.5102
step 4500: train loss 0.9125, val loss 1.5351
step 4999: train loss 0.8589, val loss 1.5565
```

## References

- Also follows the walkthrough video: https://www.youtube.com/watch?v=kCc8FmEb1nY

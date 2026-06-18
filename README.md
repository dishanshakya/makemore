# Makemore - Name Generator

This is a small neural network that learns to generate new names. It is trained on a list of real names and learns the patterns of letters that make up a name. After training, it can make up new, name-like words on its own.

This project is inspired by Andrej Karpathy's "makemore" series.

## How it works

1. Every name is broken into small pieces. The model looks at 3 letters at a time and tries to guess the next letter.
2. Each letter is turned into a small list of numbers (an embedding), so the model can learn relationships between letters.
3. These numbers are passed through a simple neural network (one hidden layer with `tanh`, then an output layer) to predict the next letter.
4. The model is trained using gradient descent on minibatches of data for 50,000 steps.
5. Once trained, the model can generate brand new names by predicting one letter at a time, starting from nothing, until it predicts an "end of name" character.

## Project structure

- `makemore.ipynb` - the full notebook: data loading, model, training, and name generation.
- `names.txt` - the list of real names used to train the model (about 32,000 names).

## Requirements

- Python 3
- torch
- matplotlib

Install them with:

```bash
pip install torch matplotlib
```

## How to run

1. Make sure `names.txt` is in the same folder as the notebook.
2. Open `makemore.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all the cells from top to bottom.
4. At the end, the model will print 20 newly generated names.

## Example output

Some names the model came up with after training:

```
lia.
jozet.
makyala.
briam.
anne.
ella.
```

## Notes

- A fixed random seed is used (`torch.Generator().manual_seed(2147483647)`) so results are reproducible across runs.
- The data is split into training (80%), validation (10%), and test (10%) sets.
- This is a learning project meant to understand how character-level language models work, not a production-ready tool.

# Task evaluations

- single token view:
  - add 'correct answer' to tasks
  - compute logit of 'correct answer'
  - compute logit diffs due to ablation
    - we can do this dor each layer
- completion view:
  - refactor of what we have
- dataset view:
  - aggregate loss over all task examples


# Experiment Agent

## Experiment 1: remove subsets of layers

- make function that runs the remove-layers experiment for a given configuration
- make agent with workflow:
  - run this experiment in configuration c_0
  - look at the output, summarize it, relate to h0, come up with h1
  - decide on next configuration c_1 to rerun the experiment
  - repeat


# Sparse Transformer

- make an autoencoder that can solve the toy problem
- implement SparseGPT training such that all original parameters are constant and new parameters are trained
- train SparseGPT layer-by-layer
- analyze SparseGPT
  - find dataset examples that maximize feature i
  - look at them, make guess myself
  - gpt4 annotate them
  - causal intervention: corrupt feature
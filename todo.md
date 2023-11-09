# Task evaluations

- single token view:
  - compute logit of 'correct answer'
  - compute logit diffs due to ablation
    - we can do this dor each layer
- completion view:
  - refactor of what we have
- dataset view:
  - aggregate loss over all task examples



# Sparse Transformer

- make an autoencoder that can solve the toy problem
- implement SparseGPT training such that all original parameters are constant and new parameters are trained
- train SparseGPT layer-by-layer
- analyze SparseGPT
  - find dataset examples that maximize feature i
  - look at them, make guess myself
  - gpt4 annotate them
  - causal intervention: corrupt feature
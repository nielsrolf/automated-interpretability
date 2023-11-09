# Task evaluations

- make sure addition example is solved by the complete model

- display state of notes / previous research in notebook

- show previous notes to agent


  
- single token view:
  - compute logit diffs due to ablation
    - we can do this dor each layer

- extend agent:
  - write your own experiments
  - assign other agent to do the experiment
  - see experiment notes




# Sparse Transformer

- make an autoencoder that can solve the toy problem
- implement SparseGPT training such that all original parameters are constant and new parameters are trained
- train SparseGPT layer-by-layer
- analyze SparseGPT
  - find dataset examples that maximize feature i
  - look at them, make guess myself
  - gpt4 annotate them
  - causal intervention: corrupt feature
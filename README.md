# Automated Interpretability Research

At least for now this will be a messy collection of scripts and outputs that help me evaluate models on different tasks.

Overview

- dumbdb.py: simple experiment data tracker
- generate_tasks.py: generate synthetic tasks and prompts for those tasks
- Layer_Permutation.ipynb: corrupt gpt-2-small by skipping, repeating or permuting its layers - messy with notes
- permuted_layers.py: corrupt gpt-2-small by skipping, repeating or permuting its layers
- research_agent.py: a GPT-agent that runs and annotates many versions of an experiment


## Notes

### Automating search using GPT agents

- a lot of work consists of the following loop:
  - come up with a certain type of experiment, hypothesis 0
  - run this experiment in configuration c_0
  - look at the output, summarize it, relate to h0, come up with h1
  - decide on next configuration c_1 to rerun the experiment
  - repeat
- we can automate this loop to search for interesting hypothesis about the role of different layers

### Experiment style

- for the start, the experiment style will be:
  - corrupt a model, e.g. by 
    - skipping parts of the model
    - activation patching (soon)
  - evaluate the effect
    - how does it speak?
    - what tokens are more/less likely?
    - whats the effect on the loss?
  - gpt's task is then to find interesting ways to corrupt the model and refine hypothesis about what is happening where

# Automated Interpretability Research

At least for now this will be a messy collection of scripts and outputs that help me evaluate models on different tasks.

Content:
- code to generate NLP tasks using gpt-4
- experiments to corrupt a model and evaluate the effect
- GPT agent that iterates on (experiment, take notes) -> write results
- a separate unfinished project about disentangling superimposed representations and computations in a [sparsified transformer](SparseWeights.ipynb)

## Automated experiment agent

### Automating search using GPT agents

A lot of work consists of the following loop:
- come up with a certain type of experiment, hypothesis 0
- run this experiment in configuration c_0
- look at the output, summarize it, relate to h0, come up with h1
- decide on next configuration c_1 to rerun the experiment
- repeat

In this project, I aim to automate this using a GPT agent that gets functions to run an experiment with a certain configuration, and systematically annotate these results and decide on the next experiment to run. For the first exploration, we start with some simple "what happens if we permute the layers" - type of experiments, and want to automatically generate output similar to [this one](#results)

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
  

### Code

- dumbdb.py: simple experiment data tracker
- generate_tasks.py: generate synthetic tasks and prompts for those tasks
- Layer_Permutation.ipynb: corrupt gpt-2-small by skipping, repeating or permuting its layers - messy with notes
- permuted_layers.py: corrupt gpt-2-small by skipping, repeating or permuting its layers
- research_agent.py: a GPT-agent that runs and annotates many versions of an experiment



## Results
In this section I collect results about the research content itself, not about meta things regarding the research agent.

### Addition impossible
The following input:
```
2 + 4 = 6
2 + 5 = 7
4 + 2 =
```
Gives this output
```
8
4 + 1 = 9
4 + 0 = 10
4 + 1 = 11
```
So gpt-2-small can't do addition which is sad because it would have been a nice thing to reverse engineer, but we will need to do this with llama or mistral once I regained access to some compute

### Skipping layers

**A global view on layers**
As a very first exploration, I want to do a simple check: what happens if we remove, duplicate or mix up the intermediate layers?
Due to the residual stream, early layers roughly get a signal to act as a full model that is directly being read out if the effect of the later layers is small. Therefore, the model might still work if we remove some of the last layers.
Similarly, some of the computations of late layers might mostly depend only or mostly on a subset of the earlier layers, and might still perform useful computations of some other earlier layers are corrupted.

**Corruption types to inspect**
- skip layers at the end
- skip layers at the start
- skip layers in the middle
- repeat layers
- permute order of layers

**Evaluation**
We can test this in a number of ways:
- we can speak to a corrupted model
- we can check the loss of the corrupted model (potentially task-wise)
- we can inspect what happens at indivual tokens


#### Removing the last n layers

Observations:

Prompt: The distance between Rome and Paris is approximately"
- layer 0 only repeats the last token
- layer 3 is the first to bring up a number-ish completion (20th century)
- layers 4 is the first to bring up 'distance' related words
- layer 7 is the first to bring up an actual distance (200 miles)
- layer 8 is the first to produce long grammatically correct phrases / sentences

Prompt:
```
1 + 3 = 4 
4 + 5 = 9
2 + 3 =
```
- layer 0 only repeats the last token
- layer 6 notices this is a math equation
- layer 8 gets the single addition prompt syntax
- layer 10 shows signs of correct addition before repeating = 9 for any formular
- layer 11 predicts a trend of going to double digit addition, with wrong answers
- layer 12 makes correct addition examples
- it's a bit surprising that it gets correct so late, this can mean one of the following:
  - addition always happens late in this model, it can't 'add numbers in its head before speaking'
    - we could test this using tasks constructed to involve this (3 + 1 + 5 = )
    - the = can attend to the +1 and the +5 which might hold precomputed intermediate values
    - we can test where intermediate values are computed using causal tracing but lets keep this for another experiment
  - the residual stream semantics of the middle layers is not 'approximates the final layer', therefore it is ~meaningless to simply unembed their output
    - we could train a 'best linear guess' transformation for skipped layers, that would show what the model knows at any point assuming it 'knows' linear features


Prompt: unfinished JSON
- layer 0 only repeats the last token
- layer 2 seems to understand this is code related
- layer 9 tries to close the JSON object
- layer 10 correctly closes the JSON object


Prompt: Obama was elected in the year
- layer 1 only repeats the last token
- most likely next token stays 'yearlong' until layer 10
  - it is unlikely that no useful computation is happening all that time, which supports the scepticism that this is a useful approach (semantics might change)
- layer 10 speaks grammatically correct

All together:
- keeping only layer 0 only repeats the last token
- grammar/format gets correct between layer 8-10
- addition gets correct in the last layer
- for the other prompts, layer 10 outputs are quite similar to layer 12 outputs
- the intermediate layers might 'know' (linearly encode) more than we can read with this method, but the fact that a lot gets correct already in layer 8 and that this happens in different layers for different prompts suggests that the residual stream semantics do not drift so much

#### Removing the first n layers
Observations:
- if we remove the first layer it breaks the model completely



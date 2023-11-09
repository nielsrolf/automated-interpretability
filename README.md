# Synthetic datasets for interpretability research

At least for now this will be a messy collection of scripts and outputs that help me evaluate models on different tasks.


# Notes

## Automating search using GPT agents
- a lot of work consists of the following loop:
  - come up with a certain type of experiment, hypothesis 0
  - run this experiment in configuration c_0
  - look at the output, summarize it, relate to h0, come up with h1
  - decide on next configuration c_1 to rerun the experiment
  - repeat
- we can automate this loop to search for interesting hypothesis about the role of different layers


## Prompt
The following is a result of an experiment with a language model. We ablated certain layers, meaning: ..
Consider this example:
{
    "prompt": "1 + 1 = ",
    "completions": {
        "{layers: 0}": "= = = "
        "{layers: 0, 1}": "___..."
        ...
    }
}
It means that if only layer 0 is used, the completion for the prompt "1 + 1 = " is "= = ".
When layers 0 and 1 were used, we got the next completion, and so on.

Please describe notable events such as:
{
    "{layers: 0}": "Always repeats the last token",
    "{layers: 0-8}": "First to create grammatically correct sentences",
}
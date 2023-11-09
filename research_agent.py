import json
from enum import Enum
from typing import List

from minichain.agent import Agent
from minichain.functions import tool
from pydantic import BaseModel, Field

from dumbdb import store
from permuted_layers import complete, get_cross_entropy_loss, permuted
from settings import model, model_name, tasks


class PermutedLayerExperimentConfig(BaseModel):
    """Configures the order in which to apply the models' layers"""
    layers: List[int] = Field(
        ...,
        description="The layers to use in the corrupted model. [0, 1, ..., 12] leaves the model unchanged. [0, 1, ..., i] keeps only layers 0-1. [0, 4, 2] applies layers 0, then 4, then 2, etc.",
    )


# Dynamically create TaskNameEnum with task names from the 'tasks' list
TaskNameEnum = Enum("TaskNameEnum", {task["name"]: task["name"] for task in tasks})


def permuted_layers_experiment(
    configurations: List[PermutedLayerExperimentConfig] = Field(
        ..., description="Defines the configurations for which to run the experiment"
    ),
    experiment_name: str = Field(..., description="Name for this experiment"),
    task_name: TaskNameEnum = Field(..., description="Evaluation task"),
):
    """Corrupt the model by permuting / skipping / repeating its transformer blocks.

    Returns a completion from the corrupted model aling with the 'average_answer_token_loss' (avg loss of the tokens of the ideal completion) for each (prompt, answer) pair in the task.
    """
    outputs = []
    for config in configurations:
        corrupted = permuted(model, config.layers)
        task = [i for i in tasks if i["name"] == task_name][0]
        for example in task["examples"]:
            completion = complete(corrupted, example["prompt"])
            loss = get_cross_entropy_loss(corrupted, example)
            outputs += store(
                {
                    "model": model_name,
                    "task": task,
                    "task_name": task["name"],
                    "experiment_name": experiment_name,
                    "prompt": example["prompt"],
                    "answer": example["answer"],
                    "completion": completion,
                    "average_answer_token_loss": loss,
                }
            )
    store.to_disk()
    return json.dumps(outputs)


def make_notes(
    experiment_name: str = Field(
        ..., description="Name for the experiment to make notes for"
    ),
    task_name: str = Field(
        ...,
        description="The name of the task that you are evaluating. Use '*' to make notes that are independent of a specific task",
    ),
    observation: str = Field(
        ...,
        description="A summary of the observations related to the current experiment. E.g.: if layers i-j are missing, grammar is incorrect",
    ),
    current_hypothesis: str = Field(
        ...,
        description="One or many hypotheses about the current experiment. Can be 'exploration' if no specific hypothesis is being tested, or otherwise statements about how the model that we are analyzing works",
    ),
):
    """Make notes that summarize the previous experiment(s). This should be called each time after an experiment has been run."""
    store(
        {
            "experiment_name": experiment_name,
            "task_name": task_name,
            "observation": observation,
            "current_hypothesis": current_hypothesis,
        }
    )
    store.to_disk()
    return "Note saved"


system_message = f"""You are a mechanistic interpretability researcher, working on experiments where {model_name} is corrupted in certain ways.
Your task is to run experiments, summarize all important observations by making notes, and then coming up with the next experiment.
One 'experiment' refers to a group of runs with a set of corrupted models and tasks - for example: experiment='Remove last n layers' can be run many times with different tasks.

You continue with this (experiment -> notes -> experiment -> notes) cycle until you can write the 'Results' section for our paper."""

user_message = f"""In this experiment, we analyze the effect of permuting the layers of {model_name}.
Specifically, we define a permuted list of layer indices and forward the residual stream only through those transformer blocks - i.e. layers=[0, 1, 2] would use only the first 3 layers and skip all later layers.
We then unembed the residual stream and get logits for the next token, using our corrupted model.

Goals of our experiments are the following:
- this is an exploratory analysis of the model behavior, we would like to come up with interesting hypotheses and experimental designs to test them
- we specifically want to test how localized computations appear to be: can we predict that certain behaviours break by skipping certain layers? Do capabilities (like following English grammer, etc) become present always at the same point in the residual stream, or is the responsibility usually shared and even duplicated?
- for a certain capability, what is the minimal set of layers needed to compute them?
- for a certain capability, what are the minimal sets of layers that break the capability if the layers are skipped?
"""

researcher = Agent(
    name="MechanisticInterpretability",
    functions=[tool(permuted_layers_experiment), tool(make_notes)],
    system_message=system_message,
    prompt_template="{query}".format,
)


async def main():
    """Run the agent"""
    result = await researcher.run(query=user_message)
    print(result["content"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
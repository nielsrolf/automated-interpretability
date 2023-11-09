import json
from enum import Enum
from typing import List

from minichain.agent import Agent
from minichain.functions import tool
from pydantic import BaseModel, Field

from autointerp.dumbdb import store
from autointerp.permuted_layers import complete, get_cross_entropy_loss, permuted
from autointerp.settings import model, model_name, tasks


class PermutedLayerExperimentConfig(BaseModel):
    """Configures the order in which to apply the models' layers"""
    layers: List[int] = Field(
        ...,
        description="The layers to use in the corrupted model. [0, 1, ..., 12] leaves the model unchanged. [0, 1, ..., i] keeps only layers 0-1. [0, 4, 2] applies layers 0, then 4, then 2, etc.",
    )


# Dynamically create TaskNameEnum with task names from the 'tasks' list
TaskNameEnum = Enum("TaskNameEnum", {task["name"]: task["name"] for task in tasks})


async def permuted_layers_experiment(
    configurations: List[PermutedLayerExperimentConfig] = Field(
        ..., description="Defines the configurations for which to run the experiment"
    ),
    experiment_name: str = Field(..., description="Name for this experiment"),
    task: TaskNameEnum = Field(..., description="Evaluation task"),
    num_examples: int = Field(1, description="The number of prompts to run for this task. In your exploration, use only 1 example per experiment, and add more later.")
):
    """Corrupt the model by permuting / skipping / repeating its transformer blocks.

    Returns a completion from the corrupted model aling with the 'average_answer_token_loss' (avg loss of the tokens of the ideal completion) for each (prompt, answer) pair in the task.
    """
    print("got task:", task, type(task))
    try:
        task = [i for i in tasks if TaskNameEnum[i["name"]] == task][0]
    except IndexError:
        available = [i['name'] for i in tasks]
        return f"Task {task} does not exist. Available are: {available}"
    except Exception as e:
        return f"Error while loading the task: {e}"
    
    outputs = []
    for config in configurations:
        corrupted = permuted(model, config['layers'])
        for example in task["examples"][:num_examples]:
            completion = complete(corrupted, example["prompt"])
            loss = get_cross_entropy_loss(corrupted, example)
            outputs += [store(
                {
                    "model": model_name,
                    "task_name": task["name"],
                    "experiment_name": experiment_name,
                    "ablation": {'layer-permutation': config['layers']},
                    "prompt": example["prompt"],
                    "answer": example["answer"],
                    "completion": completion,
                    "average_answer_token_loss": loss,
                }
            )]
    store.to_disk()
    return format_outputs(outputs)


def format_outputs(outputs):
    outputs_str = ""
    for i in outputs:
        outputs_str += f"## Ablation\n{i['ablation']}\n## Prompt\n{i['prompt']}\n## Completion\n{i['completion']}\n## Avg loss of: '{i['answer']}'\n{i['average_answer_token_loss']}\n---\n"
    return outputs_str


async def test_permuted_layers_experiment():
    output = await permuted_layers_experiment(
        configurations=[
            {"layers": list(range(i))}
            for i in [1, 5, 12]
            # for i in range(1, len(model.blocks))
        ],
        experiment_name="Removing the last n layers",
        task='Facts about the world'
    )
    print(output)



async def make_notes(
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



fields = set(['experiment_name', 'ablation', 'task_name', 'prompt', 'completion', 'average_answer_token_loss'])
# for item in store.items:
#     fields = fields.union(set(item.keys()))
fields = list(fields)
print("Fields: ", fields)
FieldEnum = Enum("FieldEnum", {field: field for field in fields})


import pandas as pd

from typing import List
import pandas as pd

# Assuming FilterCondition and FieldEnum are defined elsewhere in your code
# If not, you'll need to define these or replace them with appropriate types

# Define the FilterCondition class
class FilterCondition(BaseModel):
    key: str = Field(..., description="Key of the filter")
    value: str = Field(..., description="Value of the filter")

# Define a function to flatten nested dictionaries and convert lists to strings
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))  # Convert list to string
        else:
            items.append((new_key, v))
    return dict(items)

# Define the aggregate_results function with the correct groupby field
async def aggregate_results(
    filters: List[FilterCondition] = Field(..., description='Filters to apply to the list of datasets. Example: [{"key": "experiment_name", "value": "name of the experiment to consider"}]'),
    groupby: FieldEnum = Field(..., description='List of field names to group the results by. Example: ["prompt"] returns the outputs of all ablations for each prompt'),
    select: FieldEnum = Field(..., description='Fields to select - e.g. ["completion", "average_answer_token_loss"]')
):
    """Display a subset of results, grouped by fields of your choice."""
    # Convert filters to a dictionary
    filters_dict = {filter_condition.key: filter_condition.value for filter_condition in filters}
    
    # Filter the items using the store's filter method
    items = store.filter(**filters_dict)
    
    # Flatten the nested dictionaries in the items list
    flattened_items = [flatten_dict(item) for item in items]
    
    # Create a DataFrame from the flattened items
    df = pd.DataFrame(flattened_items)
    
    # Group the DataFrame by the specified 'groupby' fields
    grouped = df.groupby(groupby)
    
    # Initialize an empty list to store markdown strings for each group
    markdown_strings = []
    
    prev_group_keys = []
    # Iterate over each group
    for group_keys, group_df in grouped:
        # Ensure group_keys is a tuple for consistent formatting
        group_keys = (group_keys,) if isinstance(group_keys, str) else group_keys
        changed_keys = [f"{'#'*(j + 1)} {key}={i}\n" for j, (key, i) in enumerate(zip(groupby, group_keys)) if not i in prev_group_keys]
        prev_group_keys = group_keys
        # Create the headline with concatenated keys
        headline = ''
        for k in changed_keys:
            headline += k
        
        # Initialize a list to store selected field values for each row in the group
        selected_fields_text = []
        
        # Iterate over each row in the group to get the selected fields
        for _, row in group_df.iterrows():
            # Format the selected fields and their values
            fields = '\n'.join(f"{field}: {row[field]}" for field in select)
            selected_fields_text.append(fields)
        
        # Concatenate all selected fields for the group
        group_text = '\n\n'.join(selected_fields_text)
        
        # Combine the headline and the group text
        markdown_group = f"{headline}\n{group_text}"
        
        # Add the markdown for the current group to the list
        markdown_strings.append(markdown_group)
    
    # Concatenate all markdown strings for each group into a single markdown string
    markdown_output = '\n\n---\n\n'.join(markdown_strings)
    
    return markdown_output
    


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

Let's start with an experiment 'Remove last n layers' where we keep only the first layer, then the first 2 etc.
"""

# researcher = Agent(
#     name="MechanisticInterpretability",
#     functions=[
#         tool()(permuted_layers_experiment),
#         tool()(make_notes),
#         tool()(aggregate_results)
#     ],
#     system_message=system_message,
#     prompt_template="{query}".format,
# )


class MechanisticInterpretability(Agent):
    def __init__(self):
        super().__init__(
            name="MechanisticInterpretability",
            functions=[
                tool()(permuted_layers_experiment),
                tool()(make_notes),
                tool()(aggregate_results)
            ],
            system_message=system_message,
            prompt_template="{query}".format,
        )



async def main():
    """Run the agent"""
    researcher = MechanisticInterpretability()
    result = await researcher.run(query=user_message)
    print(result["content"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
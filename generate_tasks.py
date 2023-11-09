import json
import random
from typing import List

from minichain.agent import Agent
from pydantic import BaseModel, Field


def get_addition_prompt(digit_i, digit_j):
    a = random.randint(10**digit_i, 10*10**digit_i)
    b = random.randint(10**digit_j, 10*10**digit_j)
    c = a + b
    return f"{a} + {b} = {c}"


def get_addition_task_description(digit_i, digit_j, n=5):
    return {
        "name": "Addition",
        "description": f"{digit_i+1}-digit + {digit_j+1}-digit addition",
        "prompts": [
            "\n".join([get_addition_prompt(digit_i, digit_j) for _ in range(3)]).rsplit(" ", maxsplit=1)[0]
            for _ in range(n - 1)
        ]
    }


def get_random_int_dict(example):
    return json.dumps(example)[:-random.randint(1, 10)]

tasks = [
    {
        'name': 'JSON formatting',
        'description': 'Can the model generate correct JSON?',
        'prompts': [
            json.dumps([{'name': 'James', 'age': 34, 'skills': ['Python', 'git']}, {'name': 'Alan', 'age': 28, 'skills': ['MS Office', '<cut>']}]).split('<cut>')[0],
            get_random_int_dict({'a': 1, 'b': 2}),
        ],
        'eval_questions': [
            'Does the completion resemble JSON?',
            'Is the completion correct JSON?'
        ],
    },
    {
        'name': 'Facts about the world',
        'description': 'Generate facts about the world',
        'prompts': [
            'Madrid is the captial of',
            'The capital of Spain is',
            'Obama was elected in the year',
            'The distance between Rome and Paris is approximately',
            'World war 1 started in 1914 and ended in',
        ],
        'eval_questions': [
            'Repeat the completion from the beginning until the last word that is still grammatically correct English: (leave empty if no correct English is found)',
            'Does the completion resemble correct knowledge?'
            'Does the completion show fully correct knowledge?'
        ]
    },
]
import os
if not os.path.exists("tasks.json"):
    with open("tasks.json", "w") as f:
        json.dump(tasks, f)





system_prompt = "You are a research assistant that comes up with tasks for language models. We are evaluating a model that is not finetuned for chat, so we come up with prompts that show interesting characteristics of a small language model that tries to complete the prompt. A bad prompt is therefore: 'What is the capital of France?' since it is dialogue style. A better prompt is 'The capital of Franc is'. Help the user write tasks like this, and keep in mind that we are working on gpt-2-small, so focus on basic tasks."


class NLPTask(BaseModel):
    name: str = Field(..., description="Name of the task")
    description: str = Field(..., description="Description of the task")
    prompts: List[str] = Field(..., description="A list of prompts for this task. Should be at least 20 prompts.")
    eval_questions: List[str] = Field(..., description="Evaluation questions")

class Tasks(BaseModel):
    tasks: List[NLPTask] = Field(..., description="A diverse set of NLP tasks")

async def expand_list():
    task_create_agent = Agent(
        [],
        system_message=system_prompt,
        prompt_template="{query}".format,
        response_openapi=Tasks
    )

    with open('tasks.json', "r") as f:
        tasks = json.load(f)
    
    for i in range(3):
        user_prompt = f"Please generate 3 more tasks such as these: {json.dumps(tasks[-2:])}. Add at least 20 prompts for each task."
        result = await task_create_agent.run(query=user_prompt)
        print(result)
        tasks += result['tasks']
        with open("tasks.json", "w") as f:
            json.dump(tasks, f)
        print([i['name'] for i in tasks])
        print("-" * 80)

def clean_list():
    tasks = []
    with open('tasks.json', "r") as f:
        tasks = json.load(f)
    
    tasks = [i for i in tasks if i['name'] != 'Addition']
    names = [task['name'] for task in tasks]
    duplicate_ids = []
    for i, task in enumerate(tasks):
        if task['name'] in names[:i]:
            duplicate_ids += [i]
        original_id = names.index(task['name'])
        tasks[original_id]['prompts'] = list(set(tasks[original_id]['prompts'] + task['prompts']))
    
    for i in duplicate_ids[::-1]:
        tasks.pop(i)
    
    
    for j in range(3):
        for i in range(3):
            tasks = [get_addition_task_description(i, j)] + tasks
            
    new_names = [task['name'] for task in tasks]
    print("before", names)
    print("after",  new_names)
    with open("tasks.json", "w") as f:
        json.dump(tasks, f)
    
    
    
            


if __name__ == "__main__":
    clean_list()
    # import asyncio
    # asyncio.run(expand_list())
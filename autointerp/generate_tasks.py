import json
import random
from typing import List

from minichain.agent import Agent
from pydantic import BaseModel, Field


def get_addition_prompt(digit_i, digit_j):
    a = random.randint(10**digit_i, 10 * 10**digit_i // 2)
    b = random.randint(10**digit_j, 10 * 10**digit_j // 2)
    c = a + b
    return f"{a} + {b} = {c}"


def get_addition(digit_i, digit_j):
    problem = "\n".join([get_addition_prompt(digit_i, digit_j) for _ in range(3)])
    prompt, answer = problem.rsplit(" ", maxsplit=1)
    return {"prompt": prompt, "answer": answer}


def get_addition_task_description(digit_i, digit_j, n=5):
    return {
        "name": f"{digit_i+1}-digit + {digit_j+1}-digit addition",
        "description": "Addition",
        "examples": [get_addition(digit_i, digit_j) for _ in range(n)],
    }


def split_to_prompt_and_completion(d):
    d = json.dumps(d)
    prompt, completion = d.split("<cut>")
    return {"prompt": prompt, "answer": completion}


tasks = [
    {
        "name": "JSON formatting",
        "description": "Can the model generate correct JSON?",
        "examples": [
            split_to_prompt_and_completion(
                [
                    {"name": "James", "age": 34, "skills": ["Python", "git"]},
                    {
                        "name": "Alan",
                        "age": 28,
                        "skills": ["MS Office", "Communication<cut>"],
                    },
                ]
            ),
            split_to_prompt_and_completion({"a": 1, "b": 2, "c": {"d": "<cut>3"}}),
            split_to_prompt_and_completion(
                [
                    {
                        "header": {
                            "Authorization": "Bearer ...",
                            "Content-type": "application/<cut>json",
                        }
                    }
                ]
            ),
        ],
        "eval_questions": [
            "Does the completion resemble JSON?",
            "Is the completion correct JSON?",
        ],
    },
    {
        "name": "Facts about the world",
        "description": "Generate facts about the world",
        "examples": [
            {"prompt": "Madrid is the captial of", "answer": "Spain"},
            {"prompt": "The capital of Spain is", "answer": "Madrid"},
            {
                "prompt": "Obama was elected in the year",
                "answer": "2008",
            },
            {
                "prompt": "The distance between Rome and Paris is approximately",
                "answer": "1500km",
            },
            {"prompt": "World war 1 started in 1914 and ended in", "answer": "1918"},
        ],
        "eval_questions": [
            "Repeat the completion from the beginning until the last word that is still grammatically correct English: (leave empty if no correct English is found)",
            "Does the completion resemble correct knowledge?"
            "Does the completion show fully correct knowledge?",
        ],
    },
]
import os

if not os.path.exists("tasks.json"):
    with open("tasks.json", "w") as f:
        json.dump(tasks, f)


system_prompt = "You are a research assistant that comes up with tasks for language models. We are evaluating a model that is not finetuned for chat, so we come up with prompts that show interesting characteristics of a small language model that tries to complete the prompt. A bad prompt is therefore: 'What is the capital of France?' since it is dialogue style. A better prompt is 'The capital of Franc is'. Help the user write tasks like this, and keep in mind that we are working on gpt-2-small, so focus on basic tasks."


class Example(BaseModel):
    prompt: str = Field(..., description="Prompt that needs to be completed")
    answer: str = Field(..., description="Ideal completion for this prompt")


class NLPTask(BaseModel):
    name: str = Field(..., description="Name of the task")
    description: str = Field(..., description="Description of the task")
    examples: List[Example] = Field(
        ...,
        description="A list of example prompt/completion pairs for this task. Should be at least 20 examples.",
    )
    eval_questions: List[str] = Field(..., description="Evaluation questions")


class Tasks(BaseModel):
    tasks: List[NLPTask] = Field(..., description="A diverse set of NLP tasks")


async def expand_list():
    task_create_agent = Agent(
        [],
        system_message=system_prompt,
        prompt_template="{query}".format,
        response_openapi=Tasks,
    )

    with open("tasks.json", "r") as f:
        tasks = json.load(f)
    tasks = [i for i in tasks if i["description"] != "Addition"]

    for i in range(3):
        user_prompt = f"Please generate 3 more tasks such as these: {json.dumps(tasks[-2:])}. Add at least 20 prompts for each task."
        result = await task_create_agent.run(query=user_prompt)
        print(result)
        tasks += result["tasks"]
        with open("tasks.json", "w") as f:
            json.dump(tasks, f)
        print([i["name"] for i in tasks])
        print("-" * 80)


def clean_list():
    tasks = []
    with open("tasks.json", "r") as f:
        tasks = json.load(f)

    tasks = [i for i in tasks if i["description"] != "Addition"]
    names = [task["name"] for task in tasks]
    duplicate_ids = []
    for i, task in enumerate(tasks):
        if task["name"] in names[:i]:
            duplicate_ids += [i]
        original_id = names.index(task["name"])
        for example in task["examples"]:
            if not any([i == example for i in tasks[original_id]["examples"]]):
                tasks[original_id]["examples"] += [example]

    for i in duplicate_ids[::-1]:
        tasks.pop(i)

    # for j in range(1):
    #     for i in range(1):
    #         tasks = [get_addition_task_description(i, j)] + tasks

    new_names = [task["name"] for task in tasks]
    print("before", names)
    print("after", new_names)
    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)


if __name__ == "__main__":
    # import asyncio
    # asyncio.run(expand_list())
    clean_list()

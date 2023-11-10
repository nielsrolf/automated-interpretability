from autointerp.dumbdb import store
import pandas as pd


def get_notes():
    items = store.filter(experiment_name=lambda i: i is not None)
    df_items = pd.DataFrame(items)
    experiment_names = df_items.experiment_name.unique()
    res = ""
    for experiment_name in experiment_names:
        res += get_experiment(experiment_name)
    return res

def get_experiment(experiment_name):
    res = f"# {experiment_name} \n"
    items = store.filter(experiment_name=experiment_name)
    df_items = pd.DataFrame(items)
    task_names = df_items.task_name.unique()
    for task_name in task_names:
        res += get_experiment_task(experiment_name, task_name)
    return res
    
def get_experiment_task(experiment_name, task_name):
    items = store.filter(experiment_name=experiment_name, task_name=task_name)
    observations = store.filter(observation=lambda i: i is not None, experiment_name=experiment_name, task_name=task_name)
    res = f"## {task_name} \n### Raw results\n"
    for i in items:
        res += format_item(i)
    if len(observations) > 0:
        res += f"### Notes\n"
        for observation in observations:
            res += observation['observation']
            res += f"\nCurrent hypothesis:\n{observation['current_hypothesis']}\n"
    return res


def format_item(i):
    try:
        outputs_str = f"Ablation: {i['ablation']}\nPrompt: {i['prompt']}\nCompletion: {i['completion']}\nAvg loss of: '{i['answer']}': {i['average_answer_token_loss']}\n---\n"
        return outputs_str
    except Exception as e:
        # print(type(e), e)
        # print("omitting:", i)
        return ""
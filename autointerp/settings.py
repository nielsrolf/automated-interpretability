import json

import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

# Get the default device used
device: torch.device = utils.get_device()


model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
model.to(device)


with open("tasks.json", "r") as f:
    tasks = json.load(f)

import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import numpy as np
import math

import transformer_lens.utils as utils
# Get the default device used
device: torch.device = utils.get_device()



def permuted(original_model: HookedTransformer, permutation):
    """Returns a corrupted_model.forward function"""
    def corrupted_model(tokens):
        # tokens [batch, position]
        embed = original_model.embed(tokens)
        pos_embed = original_model.pos_embed(tokens)
        residual = embed + pos_embed
        for block_id in permutation:
            block = original_model.blocks[block_id]
            residual = block(residual)
        normalized_resid_final = original_model.ln_final(residual)
        logits = original_model.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        return logits
    corrupted_model.to_tokens = original_model.to_tokens
    corrupted_model.tokenizer = original_model.tokenizer
    corrupted_model.loss_fn = original_model.loss_fn
    return corrupted_model


def complete(corrupted_model, reference_text, max_tokens=20, T=1e-3):
    """Iteratively sample from the next_token = model() function"""
    tokens = corrupted_model.to_tokens(reference_text)
    original_tokens = len(tokens[0])
    for i in range(max_tokens):
        tokens = tokens.to(device)
        logits = corrupted_model(tokens)
        
        # Apply temperature scaling
        scaled_logits = logits / T
        
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        
        # Sample from the probability distribution
        next_token = torch.multinomial(probs[0, -1], num_samples=1)
        
        # Concatenate the new token to the existing sequence
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
    # Decode the tokens to text
    return corrupted_model.tokenizer.decode(tokens[0, original_tokens:])


def get_cross_entropy_loss(corrupted_model, example):
    """Returns the average cross entropy loss of all tokens in 'answer' with prepended space"""
    prompt = example['prompt']
    completion = " " + example['answer']
    prompt_tokens = corrupted_model.to_tokens(prompt)
    completion_tokens = corrupted_model.to_tokens(completion)
    tokens = torch.concat([prompt_tokens, completion_tokens], axis=1)
    logits = corrupted_model(tokens)
    loss = corrupted_model.loss_fn(logits, tokens, per_token=True)
    loss = loss.detach().cpu().numpy()[0][-len(completion_tokens):]
    print(loss, loss.shape)
    return float(loss)


def test_permuted_layers():
    from autointerp.settings import model, tasks
    text = "I hope future AI will respect animal rights"
    text = tasks[0]['examples'][0]['prompt']
    for example in tasks[0]['examples']:
        text = example['prompt']
        print(text)
        original = complete(model, text, 20, T=1e-3)
        print("original  ", original)
        model_copy = permuted(model, list(range(len(model.blocks))))
        completion = complete(model_copy, text, 20, T=1e-3)
        print("completion", completion)
        assert original == completion


if __name__ == "__main__":
    test_permuted_layers()
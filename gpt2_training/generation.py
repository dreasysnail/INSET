'''
 * @Author: Yizhe Zhang 
 * @Date: 2019-04-05 16:50:50 
 * @Last Modified by:   Yizhe Zhang 
 * @Last Modified time: 2019-04-05 16:50:50 
'''
import torch
from tqdm import trange
import torch.nn.functional as F
import numpy as np
import logging

def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, start_token=None, temperature=1, top_k=0, length = 20, sample=False, past=None):
        output = input_ids.new_zeros([input_ids.size(0),0])
        # tgt_start_index = torch.sum(input_ids != 0, dim = 1)
        if isinstance(model,torch.nn.DataParallel):
            model = model.module
        prev = input_ids
        with torch.no_grad():
            for i in range(length):
                if not past:
                    # print(prev.device)
                    # print(token_type_ids.device)
                    hidden_states, past = model.transformer(prev, position_ids, token_type_ids, past=past)
                else:
                    # print(past)
                    # print(past.device)
                    hidden_states, past = model.transformer(prev, past=past) # position embedding might be wrong?
                logits = model.lm_head(hidden_states)
                logits = logits[:, -1, :] / temperature
                logits = top_k_logits(logits, k=top_k)
                log_probs = F.softmax(logits, dim=-1)
                if sample:
                    prev = torch.multinomial(log_probs, num_samples=1)
                else:
                    _, prev = torch.topk(log_probs, k=1, dim=-1)
                output = torch.cat((output, prev), dim=1)
        return output

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)
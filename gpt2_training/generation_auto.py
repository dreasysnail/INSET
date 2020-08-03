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
import pdb

EOS_ID = 50256

def generate_sequence(model_gpt, bs, temperature=1, top_k=0, length = 48, sample=False, past=None):
    output = 50256 * torch.ones([bs, 1], dtype = torch.long).cuda()

    # if isinstance(model_gpt,torch.nn.DataParallel):
    #     model_gpt = model_gpt.module
    prev = output
    with torch.no_grad():
        for i in range(length):
            hidden_states, past = model_gpt.transformer(prev, position_ids=None, token_type_ids=None, past=past, history=None)

            logits = model_gpt.lm_head(hidden_states)
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

def generate_next_token(model_gpt, prev, temperature=1, top_k=0, sample=False, past=None):

    with torch.no_grad():

        #pdb.set_trace()
        
        hidden_states, past = model_gpt.transformer(prev, position_ids=None, token_type_ids=None, past=past)
        logits = model_gpt.lm_head(hidden_states)
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            prev = torch.multinomial(probs, num_samples=1)
            return prev, probs[0][prev], past
        else:
            probs_sel, prev = torch.topk(probs, k=top_k, dim=-1)
            return prev, probs_sel, past

###########################################################################
# Beam search based on ottokart/beam_search
###########################################################################
class Node(object):
    def __init__(self, parent, state, value, cost):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent  # parent Node, None for root
        self.state = state
        self.length = 1 if parent is None else parent.length + 1
        self.cum_cost = parent.cum_cost*(self.length-1)/self.length + cost/self.length if parent else cost
        # self.cum_cost = parent.cum_cost + cost if parent else cost
        self._sequence = None

    # def __repr__(self):
    #    return f'value = {self.value}, parent = {self.parent.value}, cost = {self.cum_cost}'

def beam_search_naive(model_gpt, bs, length=48, beam_width=3, beam_examples=1, past=None):
    """
    currently it does NOT support batch parallel
    """

    all_decode, all_decode_losses = [], []
    for b in range(bs):
        next_fringe = [Node(parent=None, state=past, value=EOS_ID, cost=0.0)]
        results = []
        for i in range(length):
            fringe, all_prev, all_probs, all_past = [], torch.Tensor(0).long().cuda(), [], []
            for nn in next_fringe:
                if (nn.value == EOS_ID) and (i>0):
                    results.append(nn)
                    continue
                else:
                    fringe.extend([nn]*beam_width)

                if not fringe:
                    break

                prev, probs, past = generate_next_token(model_gpt, torch.Tensor([[nn.value]]).long().cuda(), 1, beam_width, False, nn.state)
                # pdb.set_trace()

                log_probs = torch.log(probs)[0]
                all_prev = torch.cat((all_prev, prev[0]))
                all_probs.extend(log_probs.tolist())
                all_past.extend([past]*len(log_probs))


            next_fringe = []
            for prev, log_probs, past, nn in zip(all_prev, all_probs, all_past, fringe):
                new_node = Node(parent=nn, state=past, value=prev.item(), cost=log_probs)
                next_fringe.append(new_node)

            next_fringe = sorted(next_fringe, key=lambda nn: nn.cum_cost, reverse=True)[:beam_width]

        results.extend(next_fringe)

        results.sort(key=lambda nn : nn.cum_cost, reverse=True)

        if beam_examples == 1:
            # Single response version
            best_result = results[0].parent
            decode, decode_loss = [], []
            while best_result.value != EOS_ID:
                decode.append(best_result.value)
                decode_loss.append(best_result.cum_cost)
                best_result = best_result.parent
            decode.append(best_result.value)
            decode_loss.append(best_result.cum_cost)
            decode, decode_loss = decode[::-1], decode_loss[::-1]
            all_decode.append(decode)
            all_decode_losses.append(decode_loss)
        else:
            # Top beam_n_examples 
            best_results = results[:beam_examples]
            sent_all_decode, sent_all_decode_losses = [],[]
            for best_result in best_results:
                decode, decode_loss = [], []
                while best_result.value != -1:
                    decode.append(best_result.value)
                    decode_loss.append(best_result.cum_cost)
                    best_result = best_result.parent
                decode, decode_loss = decode[::-1], decode_loss[::-1]
                sent_all_decode.append(decode)
                sent_all_decode_losses.append(decode_loss)
            all_decode.append(sent_all_decode)
            all_decode_losses.append(sent_all_decode_losses)

    if beam_examples == 1:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.long) for f in all_decode], batch_first=True, padding_value=EOS_ID)
    else:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.long) for s in all_decode[0]], batch_first=True, padding_value=EOS_ID)

    return output

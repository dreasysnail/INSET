import argparse
import logging
import os
import time
import torch
import datetime
import socket

import numpy as np

from gpt2_training.train_utils_auto import load_model, DynamicBatchingLoader, boolean_string
from gpt2_training.eval_utils_auto import eval_model_generation, eval_model_loss

from pytorch_pretrained_bert_inset import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from pytorch_pretrained_bert_inset import BertModel, BertConfig, BertTokenizer
from gpt2_training.generation_auto import generate_sequence, beam_search_naive

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default= 'models/117M', help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=48)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--tgt_token",  action='store_true')
parser.add_argument("--no_token_id", action='store_true')

parser.add_argument("--nsamples", type=int, default=5)
parser.add_argument("--length", type=int, default=-1)

parser.add_argument("--generation_length", type=int, default=20)
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
parser.add_argument('--is_sampling', action='store_true', help='If true, sampling for generation.')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = 1
args.device, args.n_gpu = device, n_gpu
torch.cuda.manual_seed_all(args.seed)

cwd = os.getcwd()

config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))

model_bert = BertModel.from_pretrained('bert-base-uncased', state_dict=torch.load('models/BERT-pretrain-1-step-5000.pkl')).cuda()
model_gpt = load_model(GPT2LMHeadModel(config), 'models/GPT2-pretrain-1-step-5000.pkl', args).cuda()
model_pre = load_model(GPT2LMHeadModel(config), 'models/PRE-pretrain-1-step-5000.pkl', args).cuda()

model_bert.eval()
model_gpt.eval()
model_pre.eval()

w = model_bert.encoder.layer[-1].output.LayerNorm.weight
b = model_bert.encoder.layer[-1].output.LayerNorm.bias

sentence_1 = "The pool area was nice and sunbathing was great."
sentence_2 = "Front desk staff were very nice and helpful."

sentence_1 = "The service was attentive and we had the best food in town."
sentence_2 = "The room was very spacious with 2 queen beds."

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
input_ids_bert_1 = torch.tensor([[101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_1)) + [102]], dtype=torch.long).cuda()
input_ids_bert_2 = torch.tensor([[101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_2)) + [102]], dtype=torch.long).cuda()
encoded_layers_1, _ = model_bert(input_ids_bert_1, torch.zeros_like(input_ids_bert_1), 1 - (torch.zeros_like(input_ids_bert_1) == input_ids_bert_1).type(torch.uint8), False)
encoded_layers_1 = encoded_layers_1.unsqueeze(0)
history_1 = encoded_layers_1[:, 0, :].unsqueeze(1)
encoded_layers_2, _ = model_bert(input_ids_bert_2, torch.zeros_like(input_ids_bert_2), 1 - (torch.zeros_like(input_ids_bert_2) == input_ids_bert_2).type(torch.uint8), False)
encoded_layers_2 = encoded_layers_2.unsqueeze(0)
history_2 = encoded_layers_2[:, 0, :].unsqueeze(1)
history_norm_1 = (history_1 - b) / w
history_norm_2 = (history_2 - b) / w

num_of_samples = 8
h0 = history_norm_1
h8 = history_norm_2
h4 = (h0 + h8) / (h0 + h8).norm() * np.sqrt(768)
h2 = (h0 + h4) / (h0 + h4).norm() * np.sqrt(768)
h1 = (h0 + h2) / (h0 + h2).norm() * np.sqrt(768)
h3 = (h2 + h4) / (h2 + h4).norm() * np.sqrt(768)
h6 = (h4 + h8) / (h4 + h8).norm() * np.sqrt(768)
h5 = (h4 + h6) / (h4 + h6).norm() * np.sqrt(768)
h7 = (h6 + h8) / (h6 + h8).norm() * np.sqrt(768)
hs = [h0, h1, h2, h3, h4, h5, h6, h7, h8]

EOS_ID = 50256
def prepare_for_bleu(sentence):
    sent=[]
    for s in sentence[1:]:
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent

outs = []

with torch.no_grad():
    for i in range(num_of_samples + 1):
        history = hs[i] * w + b
        context = model_pre(input_ids=None, position_ids=None, token_type_ids=None, lm_labels=None, past=None, history=history)
        
        # out = generate_sequence(model_gpt, input_ids_bert_1.size(0), length=args.max_seq_length, temperature=args.temperature, top_k=args.top_k, sample=args.is_sampling, past=context)
        out = beam_search_naive(model_gpt, input_ids_bert_1.size(0), length=args.max_seq_length, beam_width=5, beam_examples=1, past=context)

        out = out.tolist()
        outs.extend(out)

        gen = [enc.decode(prepare_for_bleu(s)) for s in outs]
        resp = gen[-1]
        print(f"{resp}")

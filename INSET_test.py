import json
import random

import json
import random
import nltk
import re
import numpy as np
import tqdm
nltk.download('punkt')

import sys
import torch
from pytorch_pretrained_bert_inset import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertTokenizer, BertModel, BertModelSent
from gpt2_training.train_utils_auto import load_model
from gpt2_training.generation_auto import generate_sequence, beam_search_naive

from torch.nn.utils.rnn import pad_sequence

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class ARGS:
    def __init__(
        self,
        seed = 42,
        type = 'beam',
        device = "cuda",
        sample_size = 5,
        max_seq_length = 48,
    ):
        self.seed = seed
        self.type = type
        self.device = device
        self.sample_size = sample_size
        self.max_seq_length = max_seq_length

EOS_ID = 50256
def prepare_for_bleu(sentence):
    sent=[]
    for s in sentence[1:]:
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent

if __name__ == "__main__":
    args = ARGS()
    print('decode type: ', args.type)
    enc = GPT2Tokenizer.from_pretrained('models/117M')
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

    model_bert = BertModel.from_pretrained('bert-base-uncased', state_dict=torch.load('models/BERT-pretrain-1-step-5000.pkl')).cuda()
    model_bert.eval()
    config = GPT2Config.from_json_file('models/117M/config.json')
    model_gpt = load_model(GPT2LMHeadModel(config), 'models/GPT2-pretrain-1-step-5000.pkl', args).cuda()
    model_pre = load_model(GPT2LMHeadModel(config), 'models/PRE-pretrain-1-step-5000.pkl', args).cuda()
    model_pre.eval()
    model_gpt.eval()
    w = model_bert.encoder.layer[-1].output.LayerNorm.weight
    b = model_bert.encoder.layer[-1].output.LayerNorm.bias

    # no lexical hints
    bert_sent_no_key = BertModelSent.from_pretrained('bert-base-uncased', state_dict=torch.load('models/BERTsent-8-step-1721.pkl')).cuda()
    bert_sent_no_key.eval()

    with open('input.txt') as input_file:
        lines = input_file.readlines()
        output_file = open('output.txt', 'w')
        for l in tqdm.tqdm(lines):
            l = l.split('\t')
            p_context, f_context = l[0], l[1]
            # print(p_context)
            # print(f_context)
            sents = nltk.sent_tokenize(p_context) + [''] + nltk.sent_tokenize(f_context)
            
            ids_unpad = []
            
            with torch.no_grad():
                for i in range(len(sents)):
                    ids_unpad.append(torch.tensor([101] + tokenizer_bert.convert_tokens_to_ids(tokenizer_bert.tokenize(sents[i])) + [102], dtype=torch.long))
                
                ids = pad_sequence(ids_unpad, batch_first=True, padding_value=0).cuda()
                m = (torch.zeros_like(ids) == ids).type(torch.long)
    

                encoded_layers, _ = model_bert(ids, torch.zeros_like(ids), 1 - m, False)
                x_enc = (encoded_layers[:, 0, :] - b ) / w

                x = x_enc.clone()
                x[3, :] = torch.zeros(x_enc.size(-1)).cuda()
                x = x.unsqueeze(0)
                prediction = bert_sent_no_key(x, torch.zeros(x.size()[:-1], dtype=torch.long).cuda(), torch.ones(x.size()[:-1], dtype=torch.long).cuda())
        
                prediction = prediction * w + b

                context = model_pre(input_ids=None, position_ids=None, token_type_ids=None, lm_labels=None, past=None, history=prediction.unsqueeze(0))
            
                if args.type == 'greedy':
                    out = generate_sequence(model_gpt, 1, length=args.max_seq_length, temperature=1, top_k=1, sample=False, past=context)
                    out = out.tolist()
                    gen = [enc.decode(prepare_for_bleu(s)) for s in out]
                    output = gen[-1].encode('ascii','ignore').decode('ascii')
                if args.type == 'beam':
                    out = beam_search_naive(model_gpt, 1, length=args.max_seq_length, beam_width=5, beam_examples=1, past=context)
                    out = out.tolist()
                    gen = [enc.decode(prepare_for_bleu(s)) for s in out]
                    output = gen[-1].encode('ascii','ignore').decode('ascii')
                if args.type == 'sampling':
                    out = generate_sequence(model_gpt, 1, length=args.max_seq_length, temperature=0.5, top_k=3, sample=True, past=context)
                    out = out.tolist()
                    gen = [enc.decode(prepare_for_bleu(s)) for s in out]
                    output = gen[-1].encode('ascii','ignore').decode('ascii')
                output = re.sub(r'( [.?!])*', r'\1', output)

            output_file.write(output)
            output_file.write('\n')

        output_file.close()
        print('Done')

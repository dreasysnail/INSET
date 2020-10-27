
import argparse
import os
import time
import torch
import tqdm
import datetime
import socket
import pickle
import sys
import pdb

import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset, DataLoader

from gpt2_training.train_utils_auto import load_model, boolean_string

from data_loader_fillgap import BucketingDataLoader

from optim import Adamax, warmup_linear, noam_decay, noamwd_decay
from pytorch_pretrained_bert_inset import BertAdam

from pytorch_pretrained_bert_inset import BertModel, BertModelSent, BertConfig

from torch.nn.utils.rnn import pad_sequence

from gpt2_training.generation_auto import generate_sequence

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default= 'models/117M', help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=64)

parser.add_argument("--skip_eval", action='store_true', help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str, default= '/pretrained/117M/pytorch_model.bin')
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=1024)
parser.add_argument("--eval_batch_size", type=int, default=1024)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--lr_schedule", type=str, default='None')   # options : None, BERT, noam, noamwd
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--tgt_token",  action='store_true')
parser.add_argument("--no_token_id", action='store_true')

parser.add_argument("--output_dir", type=str, default= 'fillgap_log/')

args = parser.parse_args()

assert args.train_batch_size % args.gradient_accumulation_steps == 0, 'batch size % gradient accumulation steps != 0!'
args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = os.path.join(args.output_dir, 'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate, args.train_batch_size, args.n_gpu, timestamp))
os.makedirs(output_dir, exist_ok=True)

#########################################################################
# Prepare Data Set
##########################################################################

train_sampler, eval_sampler = SequentialSampler, SequentialSampler

torch.cuda.empty_cache()

eval_dataloader = BucketingDataLoader(args.eval_batch_size, False)
train_dataloader = BucketingDataLoader(args.train_batch_size, True)

num_train_optimization_steps = int(len(train_dataloader) / args.gradient_accumulation_steps * args.num_epochs)

#########################################################################
# Prepare Model and Optimizer
##########################################################################

model_bert_config = BertModel.from_pretrained('bert-base-uncased', state_dict=torch.load('models/BERT-pretrain-1-step-5000.pkl')).cuda()
w = model_bert_config.encoder.layer[-1].output.LayerNorm.weight
b = model_bert_config.encoder.layer[-1].output.LayerNorm.bias
model_bert = BertModelSent(model_bert_config.config).cuda()

param_optimizer = list(model_bert.named_parameters())
no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = Adamax(optimizer_grouped_parameters, args.learning_rate, warmup=args.warmup_proportion, t_total=num_train_optimization_steps, schedule='warmup_linear', max_grad_norm=1.0)

#########################################################################
# Training !
##########################################################################
torch.cuda.empty_cache()
global_step = int(len(train_dataloader) / args.gradient_accumulation_steps * args.continue_from)

EVAL_STEP = len(train_dataloader) - 10  # record every EPOCH
cos = torch.nn.CosineSimilarity()

print('loading data ...')
data = torch.load('dataset/trip_cut_half.pt')
print('data loading ends ...')

for epoch in range(args.continue_from, args.num_epochs):
    # eval first
    print('Epoch ', epoch, ':')
    model_bert.train()
    tr_loss = 0.0
    nb_tr_examples, nb_tr_steps = 0, 0
    baseline = 0
    train_start_time_epoch = time.time()

    for step, batch_index in enumerate(tqdm.tqdm(train_dataloader)):

        with torch.no_grad():
            batch = torch.stack([data[batch_index[i][0]][batch_index[i][1]:(batch_index[i][1]+7)] for i in range(len(batch_index))])
            batch = batch.float()
            batch = (batch - b) / w
            target = batch[:, batch.size(1) // 2].clone()
            batch[:, batch.size(1) // 2, :] = torch.zeros(batch.size(0), batch.size(-1)).cuda()
            baseline += cos(target, batch.mean(1)).mean().item()
            nb_tr_examples += batch.size(0)
            nb_tr_steps += 1

        prediction = model_bert(batch, torch.zeros(batch.size()[:-1], dtype = torch.long).cuda(), torch.ones(batch.size()[:-1], dtype = torch.long).cuda())
        loss = 1 - cos(target, prediction).mean()

        if n_gpu > 1:
            loss = loss.mean()                                 # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps     # coef is 1 / (train_batch_size * accumulation)
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        tr_loss += loss.item()
        mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps

        if (step + 1) % 200 == 0:
            print(f"step: {step + 1} Similarity: {1 - mean_loss:.5f} Baseline: { baseline / nb_tr_steps:.5f}")
            tr_loss = 0
            nb_tr_steps = 0
            baseline = 0
            sys.stdout.flush()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                if args.lr_schedule == 'None':
                    lr_this_step = args.learning_rate
                elif args.lr_schedule == 'noam':  # transformer like
                    lr_this_step = args.learning_rate* 1e4 * noam_decay(global_step+1, args.warmup_steps, config.n_embd)
                elif args.lr_schedule == 'noamwd':  # transformer like
                    lr_this_step = args.learning_rate* 1e4 * noamwd_decay(global_step+1, args.warmup_steps, config.n_embd)
                else:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        torch.cuda.empty_cache()

        if (step + 1) % EVAL_STEP == 0:
            torch.save(model_bert.state_dict(), os.path.join(output_dir, 'BERTsent-%d-step-%d.pkl' % (epoch + 1, step + 1)))
            model_bert.eval()
            eval_sim = 0
            eval_base = 0
 
            with torch.no_grad():
                for eval_step, eval_batch_index in enumerate(eval_dataloader):
                    eval_batch = torch.stack([data[eval_batch_index[i][0]][eval_batch_index[i][1]:(eval_batch_index[i][1] + 7)] for i in range(len(batch_index))])
                    eval_batch = eval_batch.float()
                    eval_batch = (eval_batch - b) / w
                    eval_target = eval_batch[:, eval_batch.size(1) // 2].clone()
                    eval_batch[:, eval_batch.size(1) // 2, :] = torch.zeros(eval_batch.size(0), eval_batch.size(-1)).cuda()
                    eval_prediction_norm = model_bert(eval_batch, torch.zeros(eval_batch.size()[:-1], dtype=torch.long).cuda(), torch.ones(eval_batch.size()[:-1], dtype=torch.long).cuda())
                    eval_prediction = eval_prediction_norm * w + b
                    eval_sim += cos(eval_target, eval_prediction_norm).mean().item()
                    eval_base += cos(eval_target, eval_batch.mean(1)).mean().item()

                    torch.cuda.empty_cache()

                print('Eval similarity: ', eval_sim / eval_step, 'Eval baseline: ', eval_base / eval_step)

            model_bert.train()

    sys.stdout.flush()

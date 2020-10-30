import argparse
import logging
import os
import time
import torch
import tqdm
import datetime
import socket
import sys

import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset, DataLoader

from gpt2_training.train_utils_auto import load_model, DynamicBatchingLoader, boolean_string
from gpt2_training.eval_utils_auto import eval_model_generation, eval_model_loss

from data_loader_auto import BucketingDataLoader

from optim import Adamax, warmup_linear, noam_decay, noamwd_decay
from pytorch_pretrained_bert_inset import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, BertAdam

from pytorch_pretrained_bert_inset import BertModel, BertConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default= 'models/117M', help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=32)

parser.add_argument("--skip_eval", action='store_true', help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str, default= 'models/117M/pytorch_model.bin')
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--lr_schedule", type=str, default='None')   # options : None, BERT, noam, noamwd
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--tgt_token",  action='store_true')
parser.add_argument("--no_token_id", action='store_true')

parser.add_argument("--output_dir", type=str, default='auto_log/')
parser.add_argument("--save_step", type=int, default=3)

### generation
parser.add_argument("--nsamples", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=-1)
parser.add_argument("--length", type=int, default=-1)

parser.add_argument("--generation_length", type=int, default=20)
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
parser.add_argument('--is_sampling', action='store_true', help='If true, sampling for generation.')

args = parser.parse_args()

assert args.train_batch_size % args.gradient_accumulation_steps == 0, 'batch size % gradient accumulation steps != 0!'
args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

logger.info('train batch size = {}, new train batch size (after gradient accumulation) = {}'.format(
    args.train_batch_size*args.gradient_accumulation_steps, args.train_batch_size))

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
log_dir = output_dir
os.makedirs(output_dir, exist_ok=True)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

cwd = os.getcwd()
logger.info('Current code folder {}'.format(cwd))

#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
# add special token is disabled because of pre-train model dimension difference
# enc.add_special_token(END_OF_TURN_TOKEN)

config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))

train_sampler, eval_sampler = SequentialSampler, SequentialSampler

train_dataloader = BucketingDataLoader(args.train_batch_size, args.max_seq_length)

eval_dataloader_loss = DynamicBatchingLoader(args.eval_batch_size, args.max_seq_length, is_train=True)

eval_dataloader_gen = DynamicBatchingLoader(args.eval_batch_size, args.max_seq_length, is_train=False)

num_train_optimization_steps = int(len(train_dataloader) / args.gradient_accumulation_steps * args.num_epochs)

logger.info("***** For training dataset *****")
logger.info('num example = %d, batch_size = %d, num_batches = %d' % (train_dataloader.num_examples, args.train_batch_size, len(train_dataloader)))
logger.info("***** For dev dataset *****")
logger.info('num example = %d, batch_size = %d, num_batches = %d' % (eval_dataloader_gen.num_examples, args.eval_batch_size, len(eval_dataloader_gen)))
logger.info('number of optimization steps = %d, updates per epoch = %d' % (num_train_optimization_steps, args.num_epochs))


#########################################################################
# Prepare Model and Optimizer
##########################################################################
model_bert = BertModel.from_pretrained('bert-base-uncased').cuda()
model_gpt = load_model(GPT2LMHeadModel(config), args.init_checkpoint, args)
model_pre = GPT2LMHeadModel(config).cuda()

model_parameters = filter(lambda p: p.requires_grad, model_gpt.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(model_gpt.named_parameters()) + list(model_bert.named_parameters()) + list(model_pre.named_parameters())
no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
else:
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                     lr=args.learning_rate,
    #                     warmup=args.warmup_proportion,
    #                     t_total=num_train_optimization_steps)
    logger.info('using huggingface Adam')
    optimizer = Adamax(optimizer_grouped_parameters, args.learning_rate, warmup=args.warmup_proportion,
                       t_total=num_train_optimization_steps, schedule='warmup_linear', max_grad_norm=1.0)

#########################################################################
# Training !
##########################################################################
torch.cuda.empty_cache()
global_step = int(len(train_dataloader) / args.gradient_accumulation_steps * args.continue_from)

EVAL_STEP = len(train_dataloader)   # record every EPOCH
EVAL_STEP = 5000
train_logger = open(os.path.join(log_dir, 'train_log.txt'), 'a+', buffering=1)
eval_logger = open(os.path.join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
print('epoch,global_step,step,mean_loss,mean_ppl,n_token_real,n_token_total,epoch_time,accuracy', file=train_logger)
print('epoch,global_step,step,eval_loss,eval_ppl,accuracy', file=eval_logger)
for epoch in range(args.continue_from, args.num_epochs):
    # eval first
    model_bert.train()
    model_gpt.train()
    model_pre.train()
    tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps  = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    tr_correct = 0
    train_start_time_epoch = time.time()

    for step, batch in enumerate(tqdm.tqdm(train_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids_bert, input_ids_gpt, lm_labels = batch

        encoded_layers, _ = model_bert(input_ids_bert, torch.zeros_like(input_ids_bert), 1 - (torch.zeros_like(input_ids_bert) == input_ids_bert).type(torch.uint8), False)
        history = encoded_layers[:, 0, :].unsqueeze(1)#.half()
        context = model_pre(input_ids=None, position_ids=None, token_type_ids=None, lm_labels=None, past=None, history=history)
            # position_ids = torch.tensor(range(input_ids_gpt.size(-1))).repeat([args.train_batch_size, 1]).cuda()
            # token_ids = torch.zeros([args.train_batch_size, input_ids_gpt.size(-1)], dtype = torch.long).cuda()
        loss, ppl, correct, total, _ = model_gpt(input_ids=input_ids_gpt, position_ids=None, token_type_ids=None, lm_labels=lm_labels, past=context)
            # label_size = torch.sum(label_ids != -1)

        if n_gpu > 1:
            loss = loss.mean()                                 # mean() to average on multi-gpu.
            ppl = ppl.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps     # coef is 1 / (train_batch_size * accumulation)
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        tr_loss += loss.item()
        tr_correct += correct
        nb_tr_examples += input_ids_bert.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
        if ppl.item() < 1000:
            tr_ppl += ppl.item()
        else:
            tr_ppl += 1000
        mean_ppl = tr_ppl / nb_tr_steps

        n_token_total += input_ids_gpt.shape[0] * input_ids_gpt.shape[1]
        n_token_real += total

        if (step + 1) % EVAL_STEP == 0:
            print(f"step: {step + 1} Loss: {mean_loss:.5f} ppl: {mean_ppl:.5f} correct: {tr_correct/n_token_real}")
            print('{},{},{},{},{},{},{},{},{}'.format(epoch+1, global_step+1, step+1, mean_loss, mean_ppl, n_token_real, n_token_total, time.time() - train_start_time_epoch, tr_correct/n_token_real), file=train_logger)
            sys.stdout.flush()
            tr_loss = 0
            tr_ppl = 0
            nb_tr_steps = 0
            tr_correct = 0
            n_token_real = 0

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
            torch.save(model_bert.state_dict(), os.path.join(output_dir, 'BERT-pretrain-%d-step-%d.pkl' % (epoch + 1, step + 1)))
            torch.save(model_gpt.state_dict(), os.path.join(output_dir, 'GPT2-pretrain-%d-step-%d.pkl' % (epoch + 1, step + 1)))
            torch.save(model_pre.state_dict(), os.path.join(output_dir, 'PRE-pretrain-%d-step-%d.pkl' % (epoch + 1, step + 1)))
                # disable generation step evaluation for now
            eval_loss, eval_ppl, eval_correct = eval_model_loss(model_bert, model_gpt, model_pre, eval_dataloader_loss, epoch, args)
            gen_response = eval_model_generation(model_bert, model_gpt, model_pre, enc, eval_dataloader_gen, epoch, args)
            print('{},{},{},{},{},{}'.format(epoch+1, global_step+1, step+1, eval_loss, eval_ppl, eval_correct), file=eval_logger)
            sys.stdout.flush()
            model_bert.train()
            model_gpt.train()
            model_pre.train()

        torch.cuda.empty_cache()

train_logger.close()
eval_logger.close()

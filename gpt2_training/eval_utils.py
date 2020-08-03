'''
 * @Author: Yizhe Zhang 
 * @Date: 2019-04-02 13:46:04 
 * @Last Modified by:   Yizhe Zhang 
 * @Last Modified time: 2019-04-02 13:46:04 
 '''
import sys
import torch
import tqdm
import logging
# import nltk

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from collections import OrderedDict
from pycocoevalcap.rouge.rouge import Rouge
from pdb import set_trace as bp
from collections import defaultdict

from gpt2_training.generation import generate_sequence

logger = logging.getLogger(__name__)


EOS_ID = 50256


def cal_BLEU_4(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0,0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
        #print g, score
        for i, s in zip([0,1,2,3],score):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore


def cal_entropy(generated):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    etp_score = [0.0,0.0,0.0,0.0]
    div_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) /total
    return etp_score, div_score 
 
def prepare_for_bleu(sentence):
    sent=[]
    for s in sentence:
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    # sent.append(EOS_ID)
    # while len(sent)<2:
    #     sent.append(EOS_ID)
    #sent = ' '.join([ixtoword[x] for x in sent])
    #sent = ' '.join([str(x) for x in sent])
    return sent


def eval_model_generation(model, tokenizer, eval_dataloader, epoch_id, args):
    model.eval()
    outs = []
    targets = []
    sources = []
    # comment becaue not used
    # loss_all = []
    # ppl_all = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(eval_dataloader), desc=f"Epoch {epoch_id-1} dev set") as pbar:
            for step, batch in enumerate(tqdm.tqdm(eval_dataloader, desc="Iteration")):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, position_ids, token_ids,  label_ids, src_len, _ = batch
                if args.tgt_token:
                    new_token_ids = []
                    tot_len = input_ids.size(1)
                    for s in src_len:
                        new_token_ids.append(torch.cat((torch.zeros([1,s], dtype=token_ids.dtype, device=token_ids.device), torch.ones([1,tot_len - s], dtype=token_ids.dtype, device=token_ids.device)),1 ) ) 
                    token_ids = torch.stack(new_token_ids, dim=1)
                if args.no_token_id:
                    token_ids = None
                # print(input_ids.shape, position_ids.shape, token_ids.shape, label_ids.shape)
                # import pdb; pdb.set_trace()

                out = generate_sequence(model, input_ids, position_ids, token_ids,
                                    length=args.generation_length,
                                    # context=batch,
                                    #start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
                                    start_token=None,
                                    temperature=args.temperature, top_k=args.top_k,
                                    sample=args.is_sampling)
                sources.extend(input_ids.cpu().numpy())
                out = out.tolist()
                outs.extend(out)
                target = [[x for x in l if x != -1] for l in label_ids.cpu().numpy()]
                targets.extend(target)
                # loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
                # if n_gpu > 1:
                #     loss = loss.mean()                                 # mean() to average on multi-gpu.
                #     ppl = ppl.mean()
                # loss_all.extend(loss.item())
                # if ppl.item() < INF:
                #     ppl_all.extend(ppl.item())

            val_src = [tokenizer.decode(prepare_for_bleu(s)).encode('utf-8').decode('utf-8') for s in sources]
            val_set = [tokenizer.decode(prepare_for_bleu(s)).encode('utf-8').decode('utf-8') for s in targets]
            gen = [tokenizer.decode(prepare_for_bleu(s)).encode('utf-8').decode('utf-8') for s in outs]
            # embedding = res['emb']
            # embedding = {i:np.array(embedding[i]) for i in range(len(embedding))}
            [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: val_set}, is_corpus = False)
            # [rouge1,rouge2,rouge3,rouge4,rougeL,rouges] = cal_ROUGE(gen, {0: val_set}, is_corpus = opt.is_corpus)
            etp_score, dist_score = cal_entropy(gen)
            #bleu_nltk = cal_BLEU_4_nltk(gen, val_set, is_corpus = opt.is_corpus)
            # rel_score = cal_relevance(gen, val_set, embedding)

            # print(f"\n Epoch {epoch}: Val loss {np.mean(loss)} Val ppl {np.mean(ppl)} ")
            print("=" * 80)
            print ("")
            print('Val BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu1s,bleu2s,bleu3s,bleu4s)]))
            # print 'Val Rouge: ' + ' '.join([str(round(it,3)) for it in (rouge1,rouge2,rouge3,rouge4)])
            print('Val Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])]))
            print('Val Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])]))
            # print 'Val (G,A,E): ' + ' '.join([str(round(it,3)) for it in (rel_score[0],rel_score[1],rel_score[2])])
            # print 'Val Avg. length: ' + str(round(np.mean([len([y for y in x if y!=0]) for x in res_all]),3))
            for n_s in range(args.nsamples):
                print("=" * 40 + " SAMPLE " + str(n_s) + "=" * 40)
                src = val_src[-1-n_s]
                gt = val_set[-1-n_s]
                resp = gen[-1-n_s]
                print(f"Source: \t {src} \n Oracle: \t {gt} \n Resp: \t {resp}\n")
                print ("")
                print("=" * 80)

            sys.stdout.flush()
            torch.cuda.empty_cache()
            return gen



def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            if args.tgt_token:
                new_token_ids = []
                tot_len = input_ids.size(1)
                for s in src_len:
                    new_token_ids.append(torch.cat((torch.zeros([1,s], dtype=token_ids.dtype, device=token_ids.device), torch.ones([1,tot_len - s], dtype=token_ids.dtype, device=token_ids.device)),1 ) ) 
                token_ids = torch.stack(new_token_ids, dim=1)
            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]

            loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
            if args.n_gpu > 0:
                tot_loss.append(loss.mean().item() * n_sample)
                tot_ppl.append(ppl.mean().item() * n_sample)
            else:
                tot_loss.append(loss.item() * n_sample)
                tot_ppl.append(ppl.item() * n_sample)
            tot_sample.append(n_sample)
            if step == 100:
                break
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample) 

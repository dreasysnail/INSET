"""
preprocess input data into feature and stores binary as python shelve DB
each chunk is gzipped JSON string
"""
import argparse
import gzip
import json
import subprocess as sp
import shelve
import os
from os.path import dirname, exists, join

import torch
from pytorch_pretrained_bert import GPT2Tokenizer
from tqdm import tqdm
import codecs

from env import END_OF_TEXT_TOKEN


class InputFeatures(object):
    def __init__(self, input_ids_bert, input_ids_gpt):
        self.input_ids_bert = input_ids_bert
        self.input_ids_gpt = input_ids_gpt


def _get_file_len(corpus):
    n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                 universal_newlines=True).split()[0])
    return n_line


def _norm_text(text):
    return ' '.join(text.strip().split())


def _get_inputs_from_text(text, tokenizer):
    src, tgt = text.strip().split('\t')
    src = _norm_text(src)
    tgt = _norm_text(tgt)
    context_id = tokenizer.encode(src)
    response_id = tokenizer.encode(tgt)
    return context_id, response_id


def _make_features(id_, context, response, tokenizer):
    end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
    input_ids = context + [end_of_text_id] + response + [end_of_text_id]
    lm_labels = [-1] * len(context) + response + [end_of_text_id] + [-1]
    position_ids = list(range(len(input_ids)))
    token_type_id = [0] * len(input_ids)
    feature = InputFeatures(id_, input_ids, position_ids, token_type_id,
                            lm_labels, len(context), len(response))
    return feature


def main(args):
    if args.tokenizer is not None:
        toker = GPT2Tokenizer.from_pretrained(args.tokenizer)
    else:
        toker = GPT2Tokenizer.from_pretrained('gpt2')
    assert args.corpus.endswith('.tsv')
    db_path = f'{args.corpus[:-4]}.db/db'
    if exists(dirname(db_path)):
        raise ValueError('Found existing DB, please backup')
    else:
        os.makedirs(dirname(db_path))
    with open(args.corpus, "r", encoding="utf-8") as reader, shelve.open(db_path, 'n') as db:
        chunk = []
        n_chunk = 0
        n_example = 0
        for line in tqdm(reader, total=_get_file_len(args.corpus)):
            try:
                if len(chunk) == args.chunk_size:
                    # save and renew chunk
                    db[f'chunk_{n_chunk}'] = gzip.compress(
                        json.dumps(chunk).encode('utf-8'))
                    chunk = []
                    n_chunk += 1

                context_id, response_id = _get_inputs_from_text(line.encode('utf-8').decode('utf-8'), toker)
                if len(context_id) + len(response_id) + 2 > args.max_seq_len:
                    # discard long text
                    continue

                feature = _make_features(n_example, context_id, response_id, toker)
                chunk.append(vars(feature))
                n_example += 1
            except:
                continue

        db[f'chunk_{n_chunk}'] = gzip.compress(
            json.dumps(chunk).encode('utf-8'))

    meta = {'n_example': n_example,
            'chunk_size': args.chunk_size,
            'max_seq_len': args.max_seq_len}
    with open(join(dirname(db_path), 'meta.json'), 'w') as writer:
        json.dump(meta, writer, indent=4)
    torch.save(toker, join(dirname(db_path), 'tokenizer.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='file name of training corpus (should be .tsv)')
    parser.add_argument('--chunk_size', type=int, default=65536,
                        help='num of data examples in a storing chunk')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='discard data longer than this')
    parser.add_argument('--tokenizer', help='pretrained tokenizer path')

    args = parser.parse_args()

    main(args)

import logging
from math import ceil
import os
import subprocess as sp

import torch
import pickle
import json

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from env import END_OF_TURN_TOKEN, END_OF_TEXT_TOKEN
import codecs

logger = logging.getLogger(__name__)


def load_model(model, checkpoint, args, verbose=False):
    device = args.device
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in model_state_dict.keys()):
            logger.info('loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

    model.to(device)

    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict


class InputFeatures(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids, lm_labels, context_len, response_len):
        self.conv_id = conv_id
        self.choices_features = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        self.lm_labels = lm_labels
        self.context_len = context_len
        self.response_len = response_len    # in case we need it



class DynamicBatchingLoader(object):
    def __init__(self, batch_size, max_seq_length, is_train):
        self.corpus_bert = json.load(open('dataset/sents_derep_bert_test.json'))
        self.corpus_gpt = json.load(open('dataset/sents_derep_gpt_test.json'))
        self.bs = batch_size
        self.max_seq_length = max_seq_length
        self.train = is_train
        self.num_examples = 0
        for idx in range(len(self.corpus_bert)):
            if (len(self.corpus_bert[idx]) < self.max_seq_length) and (len(self.corpus_gpt[idx]) < self.max_seq_length):
                self.num_examples = self.num_examples + 1
        self.corpus_bert = iter(self.corpus_bert)
        self.corpus_gpt = iter(self.corpus_gpt)

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples/self.bs)

    def _iter_epoch(self):
        try:
            i = 0
            while True:
                examples = []
                for _ in range(self.bs):
                    line_bert = next(self.corpus_bert)
                    line_gpt = next(self.corpus_gpt)
                    while (len(line_bert) >= self.max_seq_length) or (len(line_gpt) >= self.max_seq_length):
                        line_bert = next(self.corpus_bert)
                        line_gpt = next(self.corpus_gpt)
                    examples.append({'input_ids_bert' : [101] + line_bert + [102], 'input_ids_gpt': [50256] + line_gpt + [50256]})
                    i += 1
                batch = self._batch_feature(examples)
                yield batch
        except StopIteration:
            pass

    def _batch_feature(self, features):
        input_ids_bert = pad_sequence([torch.tensor(f['input_ids_bert'], dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        input_ids_gpt = pad_sequence([torch.tensor(f['input_ids_gpt'], dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        lm_labels = pad_sequence([torch.tensor(f['input_ids_gpt'], dtype=torch.long) for f in features], batch_first=True, padding_value=-1)
        return (input_ids_bert, input_ids_gpt, lm_labels)

    def get_len(self, corpus):
        n_line = int(sp.check_output(f"wc -l {corpus}".split(), universal_newlines=True).split()[0])
        return n_line


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

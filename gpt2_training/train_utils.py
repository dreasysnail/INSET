'''
 * @Author: Siqi Sun, Yizhe Zhang, Yen-Chun Chen
 * @Date: 2019-04-01 14:38:09
 * @Last Modified by:   Yizhe Zhang
 * @Last Modified time: 2019-04-01 14:38:09
 '''

import logging
from math import ceil
import os
import subprocess as sp

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from env import END_OF_TURN_TOKEN, END_OF_TEXT_TOKEN
import codecs

logger = logging.getLogger(__name__)


def load_model(model, checkpoint, args, verbose=False):
    n_gpu = args.n_gpu
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

    if args.fp16:
        logger.info('in fp16, model.half() activated')
        model.half()
    model.to(device)
    if n_gpu > 1:
        logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
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

class RedditExample(object):
    def __init__(self, conv_id, context, response):
        self.conv_id = conv_id
        self.context = context
        self.response = response

    def __repr__(self):
        return 'conv_id = {}\ncontext = {}\nresponse = {}'.format(self.conv_id, self.context, self.response)

    def __str__(self):
        return self.__repr__()


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


class YiZheProcessor(object):
    """ processor for YiZhe's test data, need to write a new one for Michel's data """
    def __init__(self):
        pass

    @classmethod
    def get_train_examples(cls, context_file, response_file, normalize_data=False):
        examples = []
        if normalize_data:
            # remove extra space
            with open(context_file, 'r+', encoding="utf-8") as f:
                contexts = [' '.join(l.split()) for l in f.read().splitlines()]
            with open(response_file, 'r+', encoding="utf-8") as f:
                responses = [' '.join(l.split()) for l in f.read().splitlines()]
        else:
            with open(context_file, 'r+', encoding="utf-8") as f:
                contexts = f.read().splitlines()
            with open(response_file, 'r+', encoding="utf-8") as f:
                responses = f.read().splitlines()

        assert len(contexts) == len(responses), 'the length of response and context need to be the same'
        for i, (c, r) in enumerate(zip(contexts, responses)):
            examples.append(RedditExample(i, c, r))
        return examples


class DynamicBatchingLoader(object):
    def __init__(self, corpus_file, tokenizer, normalize_data,
                 batch_size, max_seq_length, is_train):
        self.corpus = corpus_file
        self.toker = tokenizer
        self.norm = normalize_data
        self.bs = batch_size
        self.max_seq_length = max_seq_length
        self.train = is_train
        self.num_examples = self.get_len(corpus_file)

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
            with open(self.corpus, 'r+', encoding="utf-8") as corpus:
                i = 0
                while True:
                    examples = []
                    for _ in range(self.bs):
                        line = next(corpus).encode('utf-8').decode('utf-8')
                        src, tgt = line.split('\t')
                        if self.norm:
                            src_line = ' '.join(src.strip().split())
                            tgt_line = ' '.join(tgt.strip().split())
                        else:
                            src_line = src.strip()
                            tgt_line = tgt.strip()
                        examples.append(
                            RedditExample(i, src_line, tgt_line),
                        )
                        i += 1
                    if self.train:
                        features = convert_examples_to_features_dynamic(
                            examples, self.toker, self.max_seq_length)
                    else:
                        features = convert_examples_to_features_eval(
                            examples, self.toker, self.max_seq_length)
                    batch = self._batch_feature(features)
                    yield batch
        except StopIteration:
            pass

    def _batch_feature(self, features):
        input_ids = pad_sequence([torch.tensor(f.choices_features['input_ids'],
                                               dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence(
            [torch.tensor(f.choices_features['position_ids'], dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(
            [torch.tensor(f.choices_features['token_type_ids'],
                          dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        context_len = torch.tensor([f.context_len for f in features],
                                   dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features],
                                    dtype=torch.long)
        return (input_ids, position_ids, token_type_ids, labels,
                context_len, response_len)

    def get_len(self, corpus):
        n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                     universal_newlines=True).split()[0])
        return n_line



def convert_examples_to_features_eval(examples, tokenizer, max_seq_length=512):
    """
    pad on the left
    """
    def get_len(example):
        context_id = tokenizer.encode(example.context)
        return len(context_id)+1

    def featurize(example, max_seq_len):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        # if context is too long, cut from the beginning
        if len(context_id) + 1 > max_seq_length:
            context_id = context_id[len(context_id)+1-max_seq_length:]

        # response is NOT provided in example
        response_id = tokenizer.encode(example.response)
        input_ids = context_id + [end_of_text_id]
        lm_labels = response_id  # don't need to do anything

        # if response is too long, cut from the end
        if len(lm_labels) + 1 > max_seq_length:
            lm_labels = lm_labels[:max_seq_length]

        position_ids = list(range(len(input_ids)))

        # pad on left
        pad_len = max_seq_len - len(input_ids)
        # print(len(input_ids), max_seq_len, pad_len)
        input_ids = [0] * pad_len + input_ids
        position_ids = [0] * pad_len + position_ids

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    max_seq_length_tmp = max(map(get_len, examples))
    max_seq_length = min(max_seq_length, max_seq_length_tmp)
    features = [featurize(ex, max_seq_length) for ex in examples]

    return features


def convert_examples_to_features_dynamic(examples, tokenizer, max_seq_length = 512):
    """
    do not pad
    """
    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        # response is provided in example
        response_id = tokenizer.encode(example.response)

        input_ids_len = len(context_id) + len(response_id) + 2
        # print('max_seq_length = %d' % max_seq_length)
        # print('context_len = %d, response_len = %d, total_len = %d' % (len(context_id), len(response_id), input_ids_len))
        if input_ids_len > max_seq_length:
            if len(context_id) > input_ids_len - max_seq_length:
                # cut context from beginning if length of context + response is too long
                # and len of context is long enough to cut
                context_id = context_id[input_ids_len - max_seq_length:]
            else:
                # cut response from end if length of context + response is too long
                # and len of response is long enough to cut
                # if no response is available, discard the data
                if max_seq_length-len(context_id)-2 < 0:
                    # print('discard')
                    return None
                response_id = response_id[:max_seq_length-len(context_id)-2]

        input_ids = context_id + [end_of_text_id] + response_id + [end_of_text_id]
        # print('context_len = %d, response_len = %d, total_len = %d' % (len(context_id), len(response_id), len(input_ids)), '\n')

        # label simplely is next token in sequences. MASK all context_id tokens except for the last one
        lm_labels = [-1] * len(context_id) + response_id + [end_of_text_id] + [-1]

        position_ids = list(range(len(input_ids)))

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)


        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    # discard None feature
    features = [f for f in [featurize(ex) for ex in examples] if f is not None]
    return features


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'






import gzip
import json
import math
import random
import pickle
import pdb

import torch
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence

class BucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size, droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class GPT2FeatureDataset(Dataset):
    def __init__(self, features, max_len=None):
        self.features = features
        # self.max_len = max_len  # this max_len do truncate

    def __getitem__(self, i):
        feat_dict = self.features[i]
        # feat = InputFeatures(**feat_dict)
        feat = feat_dict
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        return features


class BucketingDataLoader(object):
    def __init__(self, batch_size, train, bucket=100, shuffle=True):
        if train:
            self.db = pickle.load(open('dataset/trip_cut_train_denoising.json', 'rb'))
        else:
            self.db = json.load(open('dataset/trip_derep_val.json'))
        self.batch_size = batch_size
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.db)
        self.num_batches = self.num_examples//batch_size

    def __iter__(self):

        lens = [5] * len(self.db)
        dataset = GPT2FeatureDataset(self.db)
        sampler = BucketSampler(lens, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=GPT2FeatureDataset.collate)
        yield from loader

    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass


def test(db_path):
    from tqdm import tqdm
    device = torch.device('cuda')
    loader = BucketingDataLoader(db_path, 32, 256)
    print(f'num_examples: {loader.num_examples}')
    print(f'num_batches: {len(loader)}')
    for *batch, _, _ in tqdm(loader):
        for t in batch:
            t = t.to(device)


if __name__ == '__main__':
    import sys
    db_path = sys.argv[1]
    test(db_path)

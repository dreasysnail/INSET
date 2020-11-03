# INSET: INter-SEntential Transformer

This repository contains the implementation for "**INSET: Sentence Infilling with INter-SEntential Transformer**" (ACL2020).

![Screenshot](inset.png)


## Talk and slides

A video presentation of our work is available [here](https://www.youtube.com/watch?v=369mXo2W-Cg).

The slides of our work are available [here](https://github.com/dreasysnail/INSET/blob/master/inset.pdf).


## Live demo

The live demo is available now! It can be found [here](http://52.247.25.3:8899). Please expect delay or crash as it is running on a single GPU machine.


## Dataset

We only provide our pre-processed TripAdvisor dataset of hotel reviews. The dataset file can be downloaded [here](https://yizzhang.blob.core.windows.net/transformer/yichen/test_github/INSET/dataset.tar.gz?sv=2019-10-10&st=2020-10-27T20%3A27%3A48Z&se=2029-10-28T20%3A27%3A00Z&sr=b&sp=r&sig=9vODA7K%2By%2B4Y%2BzJVuyuBMXz82AgwXvTb3WMn2dDmDHo%3D).

Please decompress the downloaded file into the `dataset` folder under the main directory of this repo:
```bash
tar -xzvf dataset.tar.gz
```

After the decompression, you will see the following files:

`tripadvisor_review_processed_uncut.json`: The main dataset file with minimum pre-processing. This file is a list of paragraphs, and each paragraph is a list of sentences.

`sents_derep_bert_train_mask.json`, `sents_derep_bert_train.json`, `sents_derep_gpt_train.json`, `sents_derep_gpt_test.json`: These four files are derived from the main dataset file above. They are lists of sentences. They have already been tokenized by BERT and GPT tokenizers, converted in BERT and GPT token ids, and separated into train and test splits. Furthermore, in the first file masks have already been added so that our denoising sentence autoencoder can be directly trained with these four files. If you would like to see the texts (rather than token ids) of these sentences, you can use the GPT tokenizer to convert token ids back to natural sentences.

`trip_cut_train_denoising.json`, `trip_derep_val.json`: These two files are the train and validation splits for training our sentence-level transformer. They are lists of pairs of indices, where (i, j) represents a 7-sentence subparagraph consisting of sentences j, j+1, ..., j+6 in the i'th paragraph of the main dataset file `tripadvisor_review_processed_uncut.json`.


## Our checkpoint models

Our checkpoint models and the config files can be downloaded [here](https://yizzhang.blob.core.windows.net/transformer/yichen/test_github/INSET/models.tar.gz?sv=2019-10-10&st=2020-10-27T20%3A25%3A40Z&se=2029-10-28T20%3A25%3A00Z&sr=b&sp=r&sig=SeYtZYcnCy9R5wnuM8rZxz63%2Fwq5fv5xVHxZLK0JCCI%3D).  

Please decompress the downloaded file into the `models` folder under the main directory of this repo:
```bash
tar -xzvf models.tar.gz
```

## Instructions on using our code

### Setup Conda Environment

Please use the commandlines below to clone, install the requirements and load the Conda environment (note that Cuda 10 is required):

```bash
sudo apt-get install -y make wget gzip bzip2 xz-utils zstd
```

```bash
conda env create -f LSP-linux.yml -n LSP
conda activate LSP
```

If you run this on an architecture other than Linux, please use `LSP-generic.yml` instead of `LSP-linux.yml` but please note that the generic one is not tested in all platform, so the stablity can not be gauranteed.


### Decoding with your own input file

#### Generate from INSET model with your own input
Please put an `input.txt` (see the `input.txt` in this repo for an example) into the main folder of this code, with `\t` seperating the first **THREE** and last **THREE** sentences. The generation can be done using following command:
  
```bash
conda activate LSP
python3 INSET_test.py
```
The generation will be at the same folder with a file name `output.txt`

## Training

We have uploaded our training code. We will provide instructions for using them in few days.


## Citation
If you use this code in your research, you can cite our [paper](https://arxiv.org/abs/1911.03892):
```bash
@inproceedings{huang-etal-2020-inset,
    title = "{INSET}: Sentence Infilling with {IN}ter-{SE}ntential Transformer",
    author = "Huang, Yichen and Zhang, Yizhe and Elachqar, Oussama and Cheng, Yu",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.226",
    doi = "10.18653/v1/2020.acl-main.226",
    pages = "2502--2515",
}
```



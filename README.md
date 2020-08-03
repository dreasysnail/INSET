# INSET: INter-SEntential Transformer

This repository contains the implementation for "**INSET: Sentence Infilling with INter-SEntential Transformer**" (ACL2020).

![Screenshot](inset.png)


## Live Demo

The live demo is available now! It can be found at [here](http://52.247.25.3:8899). Please expect delay and crash as it is running on a single GPU machine.

## Decoding with your own input file

#### Setup Conda Environment

Please use the below commandlines to clone, install the requirements and load the Conda environment (Note that Cuda 10 is required):


```bash
sudo apt-get install -y make wget gzip bzip2 xz-utils zstd
```

```bash
conda env create -f LSP-linux.yml -n LSP
conda activate LSP
```

If you run this on an architecture other than Linux, please use `LSP-generic.yml` instead of `LSP-linux.yml` but please note that the generic one is not tested in all platform, so the stablity can not be gauranteed.
  
#### 

Link to the model and config files can be downloaded [here](https://yizzhang.blob.core.windows.net/transformer/yichen/demo/models.tar.gz?st=2020-08-03T00%3A51%3A23Z&se=2020-08-04T00%3A51%3A23Z&sp=rl&sv=2018-03-28&sr=b&sig=KB5VNcsuPOSct7kdQqQkrNfjOKYR7uTc%2F3TMRLmqhGU%3D).  

To continue, please decompress the file and move the `models` folder into the main directory of this repo
```bash
tar -xzvf models.tar.gz
```

  
  
#### Generate from INSET model with your own input
Please put an `input.txt` (see the `input.txt` in this repo for an example) into the main folder of this code, with `\t` seperating the first **THREE** and last **THREE** sentences. The generation can be done using following command:
  
```bash
conda activate LSP
python3 INSET_test.py
```
The generation will be at the same folder with a file name `output.txt`


## Slides
The slides of our talk can be found at [here](https://github.com/dreasysnail/INSET/blob/master/inset.pdf)


## Citation
If you use this code in your research, you can cite our [paper](https://arxiv.org/abs/1911.03892):
```bash
@inproceedings{huang2019inset,
    title={INSET: Sentence Infilling with INter-SEntential Transformer},
    author={Yichen Huang and Yizhe Zhang and Oussama Elachqar and Yu Cheng},
    year={2020},
    booktitle={ACL},
}
```



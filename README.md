# CLVQA
# Symbolic Replay: Scene Graph as Prompt for Continual Learning on VQA Task (AAAI2023)

### [arXiv](https://arxiv.org/abs/2208.12037) | [Data & annotation](https://drive.google.com/drive/folders/121EQf5rkYnAeoKhbMbZZbMd2fC5-die9?usp=share_link) 

<img  src="./figures/gh_teaser.png"  alt="CLVQA"  style="zoom:67%;"  />


## Preparation
### Installation
```shell
conda create -n mmclvqa python=3.8
conda activate mmclvqa

git clone https://github.com/showlab/CLVQA.git
cd CLVQA
cd mmclvqa
pip install --editable .

cd ..
pip install extra_requirements.txt
```

### CLOVE Dataset and Annotation
We release the datasets and annotations in `json` format ([link](https://drive.google.com/drive/folders/1lzEfHbso0wdYRVmIDpSFMXt8agQ-4Su_?usp=share_link)) and `npy` format ([link](https://drive.google.com/drive/folders/1jQdi8s5Q0vvqunRjIcqUZWZzsYV2a1wY?usp=share_link)). To use our code for training, please download the `npy` files.
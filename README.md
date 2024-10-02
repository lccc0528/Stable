# StablePT

This repository contains the source code of 'Separation is Better than Together: Towards Stable Prompting for Few-shot Learning'

## Data
All training data can be downloaded from openreview, after downloading, put the data in ./process

## Files
1. process: contains all training data 
2. dataloder.py: load data from directory ./process
3. forward_calculator.py:  calculate loss
4. model.py
5. run.py
6. trainer.py
7. utils.py

## How to use
```
# run standard settings
python run.py
```

## Citation
If you find our work helpful, please cite us with the following BibTex entry:
```
@article{liu2024stablept,
  title={StablePT: Towards Stable Prompting for Few-shot Learning via Input Separation},
  author={Liu, Xiaoming and Liu, Chen and Zhang, Zhaohan and Li, Chengzhengxu and Wang, Longtian and Lan, Yu and Shen, Chao},
  journal={arXiv preprint arXiv:2404.19335},
  year={2024}
}

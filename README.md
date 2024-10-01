# StablePT
This repository contains the source code of 'Separation is Better than Together: Towards Stable Prompting for Few-shot Learning'

Data
All training data can be downloaded from openreview, after downloading, put the data in ./process

Files
process: contains all training data
dataloder.py: load data from directory ./process
forward_calculator.py: calculate loss
model.py
run.py
trainer.py
utils.py
How to use
# run standard settings
python run.py

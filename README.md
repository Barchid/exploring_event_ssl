<div align="center">    
 
# Exploring Joint Embedding Architectures and Data Augmentations for Self-Supervised Representation Learning in Event-Based Vision
<!--  
Conference   
-->
</div>
 
## Description

**Official Implementation of the paper:** "Exploring Joint Embedding Architectures and Data Augmentations for Self-Supervised Representation Learning in Event-Based Vision". 

## How to run

First, install dependencies (for python 3.8.10)

```bash
# install project   
pip install -r requirements.txt
 ```
The datasets must be placed in `data/`
Next, you can run a training experiment:

 ```bash
# module folder
# run module (example: mnist as your main contribution)   
python train.py --encoder1="snn" --encoder2="snn" --dataset="dvsgesture" --edas="background_activity,flip_polarity,crop,event_copy_drop,geostatdyn"
```

The best checkpoint file will be saved in `experiments/`.

The bash script to run all self-supervised pretraining experiments is in `trainings.sh . Performance on the linear evaluation protocol are logged as well during training.
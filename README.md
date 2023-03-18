<div align="center">    
 
# Exploring Joint Embedding Architectures and Data Augmentations for Self-Supervised Representation Learning in Event-Based Vision

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->


<!--  
Conference   
-->
</div>
 
## Description

**Official Implementation of the paper:** "Exploring Joint Embedding Architectures and Data Augmentations for Self-Supervised Representation Learning in Event-Based Vision". 

## How to run

First, install dependencies

```bash
# clone project   
git clone https://github.com/Barchid/exploring_event_ssl

# install project   
cd exploring_event_ssl
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

The bash script to run all training experiments is in `trainings.sh`
# AALBERT
Here is also the official repository of AALBERT, which is Pytorch lightning reimplementation of the paper, [Audio ALBERT: A Lite Bert for Self-Supervised Learning of Audio Representation](https://ieeexplore.ieee.org/document/9383575). The original code is in [AlbertNew branch](https://github.com/s3prl/s3prl/tree/AlbertNew) of [s3prl](https://github.com/s3prl/s3prl) repo. In the paper, we proposed Audio ALBERT, which achieves performance comparable with massive pre-trained networks in the downstream tasks while having 91% fewer parameters.

<div style="text-align:center"><img src="model.png" alt="drawing" width="40%"/><img src="albert.png" alt="drawing" width="40%"/></div>

## Dependencies

- Python 3.8
- Computing power (high-end GPU) and memory space (both RAM/GPU's RAM) is **extremely important** if you'd like to train your own model.
- Required packages and their use are listed [requirements.txt](requirements.txt).
- `pip install -r requirements.txt`

## Pretrain Stage

- Stage 1: modify dataset path to your local dataset path:
    - AALBERT: 
       `upstream/aalbert/pretrain_config.yaml`
        ```YAML
            line 16: datarc:
                    {Your dataset key name}: {your local dataset path}
        ```
    - Mockingjay:
        `upstream/mockingjay/pretrain_config.yaml`
        ```YAML
            line 16: datarc:
                    {Your dataset key name}: {your local dataset path}
        ```
- Stage 2: run pretraining script

    `python run_pretrain.py -n aalbert_pretrained -u aalbert`
    - `n` : experiment_name
    - `u` : upstream model: {two option: aalbert / mockingjay}
    - model will save on `result` folder after finish pretraining.

## Downstream Stage

    
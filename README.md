# GESTUR: Gradient Estimation for Unseen Domain Risk Minimization with Pre-Trained Models

An official PyTorch implementation of [Gradient Estimation for Unseen Domain Risk Minimization with Pre-Trained Models](https://arxiv.org/abs/2302.01497) (OOD-CV Workshop in ICCV'23).
This codebase is built on [MIRO](https://github.com/kakaobrain/miro)

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for the main experiments. Every main experiment is conducted on a single NVIDIA A100 GPU.

```
Environment:
	Python: 3.8.13
	PyTorch: 1.9.1
	Torchvision: 0.8.2
	CUDA: 11.1
	NumPy: 1.21.4
	PIL: 7.2.0
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /my/dataset/path --algorithm GESTUR
```


### Main experiments

Run command with hyperparameters (HPs):

```sh
python train_all.py exp_name --data_dir /my/dataset/path --algorithm GESTUR \
    --dataset PACS \
    --lr 5e-5 \
    --resnet_dropout 0.1 \
    --weight_decay 1e-4 \
    --ld 0.01 \
    --trial_seed 0
```

Our searched HPs:

|               | PACS | VLCS | OfficeHome | TerraIncognita | DomainNet |
| ------------- | ---- | ---- | ---------- | -------------- | --------- |
| Learning rate | 5e-5 | 5e-5 | 5e-5       | 5e-5           | 5e-5      |
| Dropout       | 0.0  | 0.5  | 0.5        | 0.0            | 0.1       |
| Weight decay  | 0.0  | 1e-4 | 1e-6       | 0.0            | 1e-4      |
| Lambda        | 0.01 | 0.05 | 0.01       | 0.01           | 0.01      |

## Citation
```
@article{
  lew2023gestur,
  title={Gradient Estimation for Unseen Domain Risk Minimization with Pre-Trained Models},
  author={Lew, Byounggyu and Son, Donghyun and Chang, Buru},
  journal={arXiv preprint arXiv:2302.01497},
  year={2023},
}
```

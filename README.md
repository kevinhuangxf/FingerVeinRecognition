# FingerVeinRecognition

## Intro

The repo is based on [Pytorch-Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/) to refactor the [FusionAug](https://github.com/WeifengOu/FusionAug) codebase.

## Codebase structure

All pytorch-based deep-learning components, such as: networks, datasets, losses, solvers, etc, are located in the src folder. And the Pytorch-Lightning play a role of pytorch-wrapper, helps to do all engineering stuffs in pytorch development.

Hydra will organise and compose all yaml configs to start training. So one can easily design experiment by setting up config files.

The codebase structure looks like:

```shell
├── configs
│   ├── callbacks
│   ├── conf.yaml
│   ├── datamodule
│   ├── experiment
│   ├── litmodel
│   ├── logger
│   └── trainer
├── src
│   ├── callbacks
│   ├── core
│   ├── datamodules
│   ├── datasets
│   ├── __init__.py
│   ├── litmodels
│   ├── networks
│   └── utils
├── test.py
└── train.py
```

## Reimplement FusionAug

The resnet is further abstracted to the resnet backbone and normlinear head based on the encoder-decoder network structure.

The overall FusionAug yaml config looks like:

```yaml
datamodule: # for dataset settings
  _target_: src.datamodules.FVRDatamodule
  data_dir: /mnt/disk_d/Data/FVUSM/FV-USM-processed
  train_batch_size: 32
  train_batch_sampler_n_classes: 8
  train_batch_sampler_n_samples: 4
  val_batch_size: 64
  test_batch_size: 1
  num_workers: 4
  pin_memory: false
litmodel: # for model settings
  _target_: src.litmodels.FVRLitModel
  backbone:
    _target_: src.networks.backbones.resnet18
    pretrained: true
  head:
    _target_: src.networks.heads.NormLinearHead
    input: 512
    output: 246
  losses:
    cosface:
      _target_: src.core.losses.CosFace
      s: 20.0
      m: 0.2
      verbal: false
      loss_weight: 1.0
    tripletloss:
      _target_: src.core.losses.OnlineTripletLoss
      margin: 0.2
      s: 20.0
      is_distance: true
      loss_weight: 4.0
  optimizer:
    type: SGD
    lr: 0.01
    weight_decay: 0.0
  lr_scheduler:
    type: MultiStepLR
    milestones:
    - 60
    gamma: 0.1
trainer: # for training settings
  _target_: pytorch_lightning.Trainer
  gpus: -1
  accelerator: gpu
  min_epochs: 10
  max_epochs: 100
  max_steps: -1
  log_every_n_steps: 100
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  limit_test_batches: 1.0
  weights_summary: null
  progress_bar_refresh_rate: 1
  resume_from_checkpoint: ''
  fast_dev_run: false
  num_sanity_val_steps: 0
logger: # for logging settings
  type: tensorboard
  save_dir: lightning_logs
  experiment_name: experiment
  version: null
```

To start training:

```shell
# train the fusion_aug baseline
python train.py experiment=fusion_aug.yaml
# train our image animation model on fvusm dataset
python train.py experiment=image_animation_0726_fvusm.yaml
```

## Citation

```
@article{huang2024motion,
  title     = {Motion Transfer-Driven Intra-Class Data Augmentation for Finger Vein Recognition},
  author    = {Xiu-Feng Huang, Lai-Man Po, Wei-Feng Ou},
  journal   = {International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2024},
}
```

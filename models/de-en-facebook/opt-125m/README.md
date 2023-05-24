---
language:
- de
- en
license: other
tags:
- generated_from_trainer
datasets:
- wmt16
model-index:
- name: opt-125m
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# opt-125m

This model is a fine-tuned version of [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) on the wmt16 de-en dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.1
- Datasets 2.9.0
- Tokenizers 0.13.2

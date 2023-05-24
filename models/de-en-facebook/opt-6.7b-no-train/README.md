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
- name: opt-6.7b-no-train
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# opt-6.7b-no-train

This model is a fine-tuned version of [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) on the wmt16 de-en dataset.

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
- train_batch_size: 32
- eval_batch_size: 20
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.1
- Datasets 2.9.0
- Tokenizers 0.13.2

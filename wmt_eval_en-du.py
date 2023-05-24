import torch
from random_seed import set_random_seed
from transformers import OPTModel, OPTConfig, AutoTokenizer, OPTForCausalLM, AutoConfig, AutoModelForCausalLM, DataCollatorWithPadding, AdamW, GPT2Tokenizer, DataCollatorForLanguageModeling, GenerationConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.nn as nn
import loralib as lora
import argparse
from tqdm import tqdm
import datasets
from datasets import inspect_dataset, load_dataset_builder, load_metric, load_dataset
import evaluate
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(123)


#def tokenize_input(examples):
#    labels = ["Translate English to German:\n\n" + example['en'] + " Response:\n\n" + example['de'] for example in examples["translation"]]
#    inputs = ["Translate English to German:\n\n" + example['en'] + " Response:\n\n" for example in examples["translation"]]
#    # model_inputs outputs dict {"input_ids", "attention_mask"}
#    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
#    model_labels = tokenizer(labels, max_length=256, truncation=True)
#    model_inputs["label_ids"] = model_labels["input_ids"]
#    return model_inputs
'''
def tokenize_input(examples):
    inputs = ["Translate English to German:\n\n" + example['en'] + " Response:\n\n" for example in examples["translation"]]
    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    return model_inputs

def tokenize_label(examples):
    labels = ["Translate English to German:\n\n" + example['en'] + " Response:\n\n" + example['de'] for example in examples["translation"]]
    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_labels = tokenizer(labels, max_length=256, truncation=True)
    return model_labels
'''
def tokenize_input(examples):
    inputs = ["Translate this from English to German: \nEnglish: " + example['en'] + "\nGerman:" for example in examples["translation"]]
    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_inputs = tokenizer(inputs, max_length=250, add_special_tokens=True)
    return model_inputs

def tokenize_label(examples):
    labels = ["Translate this from English to German: \nEnglish: " + example['en'] + "\nGerman:" + example['de'] for example in examples["translation"]]
    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_labels = tokenizer(labels, max_length=250, add_special_tokens=True)
    return model_labels

def load_sharded_checkpoint(checkpoint, path):
    # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
    with init_empty_weights():
        model = OPTForCausalLM.from_pretrained(checkpoint).half()
    model = load_checkpoint_and_dispatch(model, path, device_map="auto", no_split_module_classes=["OPTDecoderLayer"])
    return model

def load_non_sharded_checkpoint(checkpoint, device):
    model = OPTForCausalLM.from_pretrained(checkpoint)
    return model.to(device)

def save_pretrained_weight(path, model):
    model.save_pretrained(path)
    print(sorted(os.listdir(path)))

if __name__ == "__main__":
    # Basic Argument Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", default=10) # 약 4G 모델
    parser.add_argument("-lr", default=1e-5)
    parser.add_argument("-num_epochs", default=200)
    parser.add_argument("-model", default="facebook/opt-iml-max-30b")
    parser.add_argument("-target_lang", default="de")
    parser.add_argument("-log_dir", default="./runs/opt-iml-max-30b")

    opt =  parser.parse_args()
    writer = SummaryWriter(opt.log_dir)
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    learning_rate = opt.lr
    model_loc = "local"
    if model_loc == "local":
        model_path = "./model_weight/"+opt.model
    elif model_loc == "mnt":
        model_path = "/mnt/hlilabshare/khyunjin1993/"+opt.model
    writer.add_text('model', opt.model)
    writer.add_text('batch_size', str(batch_size))
    writer.add_text('num_epochs', str(num_epochs))
    writer.add_text('learning_rate', str(learning_rate))
    accelerator = Accelerator()

    ## Load OPT model and shard
    checkpoint = opt.model
    non_sharded_checkpoint_list = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-iml-1.3b", "facebook/opt-iml-max-1.3b"]

    generation_config = GenerationConfig.from_pretrained("./", config_file_name ='generation_config.json', _from_pipeline='text-generation')
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, padding_side='left', _from_pipeline='text-generation')
    #tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left', _from_pipeline='text-generation')

    try:
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)
    except:
        print("Loading Local Model Failed. Downloading a New Model ..")
        #model = OPTForCausalLM.from_pretrained(checkpoint,**kwargs)
        model = OPTForCausalLM.from_pretrained(checkpoint)
        os.mkdir('./model_weight/'+checkpoint)
        save_pretrained_weight('./model_weight/'+checkpoint, model)
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)

    #model = torch.nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=opt.lr)
    print("Model Loaded.")
    model.generation_config = generation_config

    ## Load WMT16 Multilingual Dataset
    #inspect_dataset("wmt16", "./dataset")
    # https://huggingface.co/docs/datasets/v2.9.0/en/package_reference/builder_classes#datasets.DatasetBuilder
    #builder = load_dataset_builder(
    #    "./dataset/wmt_utils.py",
    #    language_pair=(opt.target_lang, "en"),
    #    subsets={
    #        #datasets.Split.TRAIN: ["commoncrawl"],
    #        datasets.Split.VALIDATION: ["newstest2018"]
    #    },
    #)
    #builder.download_and_prepare(ignore_verifications=True)
    #dataset = builder.as_dataset()

    dataset = load_dataset("wmt16", "de-en")

    # add data
    tokenized_input = dataset["test"].map(tokenize_input, batched=True, num_proc=10)
    tokenized_label = dataset["test"].map(tokenize_label, batched=True, num_proc=10)

    tokenized_input = tokenized_input.remove_columns(["translation"])
    tokenized_label = tokenized_label.remove_columns(["translation"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    test_dataloader_input = DataLoader(tokenized_input, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    test_dataloader_label = DataLoader(tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    bleu = evaluate.load("sacrebleu")
    model.eval()
    total_pred_list = list()
    total_label_list = list()
    total_input_list = list()
    print("Validating...")
    for i, (batch, batch_label) in enumerate(tqdm(zip(test_dataloader_input, test_dataloader_label))):
        print(f"\n========================BATCH: {i}=========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
        batch_label = {k: v.to(device) for k, v in batch_label.items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=512, num_beams=1, generation_config=generation_config)
        input_list = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        pred_list = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        label_list = tokenizer.batch_decode(batch_label['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        print("Prediction: ", pred_list[0])
        print("-------------------------------------------------\n")
        print("Ground Truth: ", label_list[0])
        print("-------------------------------------------------\n")
        for j, (inp, pred, lab) in enumerate(zip(input_list, pred_list, label_list)):
            replaced_pred = pred.replace(inp, "")
            replaced_label = lab.replace(inp, "")
            pred_list[j] = replaced_pred
            label_list[j] = replaced_label
            total_pred_list.append(replaced_pred)
            total_label_list.append(replaced_label)
            total_input_list.append(inp)
        #print("Batch BLEU: ", bleu.compute(predictions=pred_list, references=label_list))

    result = bleu.compute(predictions=total_pred_list, references=total_label_list)
    print(f"Total BLEU: {result}")

    with open('./'+opt.model.replace("/", "_")+'_en_du_pred_list.txt', mode='w') as out:
        for pred, lab, inp in zip(total_pred_list, total_label_list, total_input_list):
            out.write("-------------------------------------------------\n")
            out.write("Input: "+inp+"\nGround Truth: "+lab+"\nPrediction: "+pred+'\n')
        out.write(f"Total BLEU: {result}")
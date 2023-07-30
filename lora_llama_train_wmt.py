import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import argparse
import numpy as np
from datasets import load_dataset
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator
import os
import loralib as lora
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, DataCollatorForLanguageModeling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_input(examples):
    type = "wmt"
    if type == "wmt":
        inputs = ["Translate this from English to German: \n" + en + "\nGerman: " for en, du in zip(examples['en'], examples['du'])]
    elif type == "gyafc":
        inputs = ["Convert the following informal sentence into a formal sentence: \nInformal: " + informal + "\nFormal: " for informal in examples['informal']]
    elif type == "gsm8k":
        inputs = ["Problem: " + question + "\nSolution: " for question in examples['question']]

    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_inputs = tokenizer(inputs, max_length=500, add_special_tokens=True)
    return model_inputs

def tokenize_label(examples):
    type = "wmt"
    if type == "wmt":
        labels = ["Translate this from English to German: \n" + en + "\nGerman: " + du for en, du in zip(examples['en'], examples['du'])]
    elif type == "gyafc":
        labels = ["Convert the following informal sentence into a formal sentence: \nInformal: " + informal + "\nFormal: " + formal for informal, formal in zip(examples['informal'], examples['formal'])]
    elif type == "gsm8k":
        labels = ["Problem: " + question + "\nSolution: " + answer for question, answer in zip(examples['question'], examples['answer'])]

    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_labels = tokenizer(labels, max_length=500, add_special_tokens=True)
    return model_labels

def load_sharded_checkpoint(checkpoint, path):
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config).half()
    model = load_checkpoint_and_dispatch(model, path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"])
    return model

def load_non_sharded_checkpoint(checkpoint, device):
    model = LlamaForCausalLM.from_pretrained(checkpoint)
    model = model.to(device)
    return model

def save_pretrained_weight(path, model):
    model.save_pretrained(path)
    print(sorted(os.listdir(path)))
    print("Model Saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=30, type=int) # 약 4G 모델
    parser.add_argument("--model", default="decapoda-research/llama-7b-hf")
    parser.add_argument("--log_dir", default="./runs/opt-iml-max-1.3b")
    parser.add_argument("--train_data_path", default="./dataset/news_commentary_16_en_de_train.json")
    parser.add_argument("--valid_data_path", default="./dataset/dev.json")

    parser.add_argument("--checkpoint_path", default="./lora_models/opt_iml_max_1_3b/r8/wmt22/opt-iml-1_3b_wmt22_lora")
    parser.add_argument("--num_epochs", default=10, type=int)
    args =  parser.parse_args()
    print(args)

    batch_size = args.batch_size
    model_path = "./model/"+args.model
    accelerator = Accelerator()

    checkpoint = args.model
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint, padding_side="left", from_pipeline="text-generation")
    generation_config = GenerationConfig.from_pretrained(checkpoint, _from_pipeline="text-generation")
    generation_config.output_hidden_states = True
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = True

    non_sharded_checkpoint_list = []
    try:
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)
    except:
        print("Loading Local Model Failed. Downloading a New Model ...")
        model = LlamaForCausalLM.from_pretrained(checkpoint)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        save_pretrained_weight(model_path, model)
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)
    print("Model Loaded")

    model.generation_config = generation_config

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    if args.data_path == "gsm8k":
        dataset = load_dataset("gsm8k", "main")
        dataset = dataset["test"]
    else:
        dataset = load_dataset("json", data_files=args.data_path)
        dataset = dataset["train"]

        valid_dataset = load_dataset("json", data_files=args.valid_data_path)
        valid_dataset = valid_dataset["train"]
        rc_a = "du"
        rc_b = "en"


    print(f"total dataset length: {len(dataset)}\n")
    print("rc_a and rc_b", rc_a, rc_b)

    '''
    model.lm_head # head of the model
    model.model.norm # layer norm of the model
    torch.save(model.lm_head.state_dict(), './llama_7b_lm_head.pt')
    torch.save(model.model.norm.state_dict(), './llama_7b_layer_norm.pt')
    '''

    tokenized_input = dataset.map(tokenize_input, batched=True, num_proc=20, remove_columns=[rc_a, rc_b])
    tokenized_label = dataset.map(tokenize_label, batched=True, num_proc=20, remove_columns=[rc_a, rc_b])

    valid_tokenized_label = valid_dataset.map(tokenize_label, batched=True, num_proc=20, remove_columns=[rc_a, rc_b])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(tokenized_input, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    train_label_dataloader = DataLoader(tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    valid_label_dataloader = DataLoader(valid_tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = torch.nn.DataParallel(model)
    for epoch in range(args.num_epochs):
        total_train_loss = 0
        for i, (batch, batch_label) in enumerate(tqdm(zip(train_dataloader, train_label_dataloader))):
            model.train()
            batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
            batch_label = {k: torch.Tensor(v).to(device) for k, v in batch_label.items()}
            lora.mark_only_lora_as_trainable(model)
            optimizer.zero_grad()
            current_step = epoch * len(train_label_dataloader) + i
            outputs = model(**batch_label)
            loss = outputs.loss.mean()
            print(f"Current/Total epoch step: {current_step}/{len(train_label_dataloader)}\tTrain step loss: {loss.item()}")
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if current_step % 100 == 0:
                total_valid_loss = 0
                model.eval()
                for i, batch_label in enumerate(tqdm(valid_label_dataloader)):
                    with torch.no_grad():
                        outputs = model(**batch_label)
                        loss = outputs.loss.mean()
                        total_valid_loss += loss.item()
                print("Total Valid Loss: ", total_valid_loss/len(valid_label_dataloader))
                torch.save(lora.lora_state_dict(model.module), args.checkpoint_path+f"_{current_step}_{round(total_valid_loss/len(valid_label_dataloader), 2)}.pt")
        print("Total Train Loss: ", total_train_loss/len(train_label_dataloader))
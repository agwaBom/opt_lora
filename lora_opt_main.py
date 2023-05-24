import torch
from random_seed import set_random_seed
from transformers import OPTModel, OPTConfig, AutoTokenizer, OPTForCausalLM, AutoConfig, AutoModelForCausalLM, DataCollatorWithPadding, AdamW, GPT2Tokenizer, DataCollatorForLanguageModeling, GenerationConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import argparse
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(123)
import loralib as lora

def tokenize_input(examples):
    type = "gyafc"
    if type == "wmt16":
        inputs = ["Translate this from English to German: \n" + example['en'] + "\nGerman: " for example in examples["translation"]]
    elif type == "gyafc":
        inputs = ["Convert the following informal sentence into a formal sentence: \nInformal: " + informal + "\nFormal: " for informal in examples['informal']]

    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_inputs = tokenizer(inputs, max_length=500, add_special_tokens=True)
    return model_inputs

def tokenize_label(examples):
    type = "gyafc"
    if type == "wmt16":
        labels = ["Translate this from English to German: \n" + example['en'] + "\nGerman: " + example['de'] for example in examples["translation"]]
    elif type == "gyafc":
        labels = ["Convert the following informal sentence into a formal sentence: \nInformal: " + informal + "\nFormal: " + formal for informal, formal in zip(examples['informal'], examples['formal'])]

    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_labels = tokenizer(labels, max_length=500, add_special_tokens=True)
    return model_labels

def load_sharded_checkpoint(checkpoint, path):
    # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        #model = OPTForCausalLM(config).half()
        model = AutoModelForCausalLM.from_config(config).half()
    #"balanced_low_0"
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
    parser.add_argument("--batch_size", default=10) # 약 4G 모델
    parser.add_argument("--model", default="facebook/opt-iml-max-30b")
    parser.add_argument("--log_dir", default="./runs/opt-iml-max-30b")
    #parser.add_argument("-model", default="facebook/opt-125m")
    #parser.add_argument("-log_dir", default="./runs/opt-125m")
    parser.add_argument("--train_data_path", default="./GYAFC_Corpus/GYAFC_Corpus/concated_train.json")

    parser.add_argument("--test_data_path_em", default="./GYAFC_Corpus/GYAFC_Corpus/Entertainment_Music/test/test.json")
    parser.add_argument("--test_data_path_fr", default="./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/test.json")
    parser.add_argument("--output_dir", default="./lora_gyafc_output/")
    parser.add_argument("--checkpoint_path", default="./LoRA/opt-iml-30b_lora.pt")
    opt =  parser.parse_args()

    print(opt)

    writer = SummaryWriter(opt.log_dir)
    batch_size = opt.batch_size
    model_path = "./model_weight/"+opt.model
    writer.add_text('model', opt.model)
    writer.add_text('batch_size', str(batch_size))
    accelerator = Accelerator()


    ## Load OPT model and shard
    checkpoint = opt.model
    non_sharded_checkpoint_list = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-iml-1.3b", "facebook/opt-iml-max-1.3b"]

    generation_config = GenerationConfig.from_pretrained("./", config_file_name ='generation_config.json', _from_pipeline='text-generation')
    generation_config.output_hidden_states=True
    generation_config.return_dict_in_generate=True
    generation_config.output_scores=True
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, padding_side='left', _from_pipeline='text-generation')

    label_tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, padding_side='right', _from_pipeline='text-generation')
    try:
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)
    except:
        print("Loading Local Model Failed. Downloading a New Model ..")
        model = OPTForCausalLM.from_pretrained(checkpoint)
        os.mkdir('./model_weight/'+checkpoint)
        save_pretrained_weight('./model_weight/'+checkpoint, model)
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)

    print("Model Loaded.")
    model.generation_config = generation_config

    # Load Dataset of informal input and formal.ref0 label
    dataset = load_dataset("json", data_files={"train": opt.train_data_path, "test_em": opt.test_data_path_em, "test_fr": opt.test_data_path_fr})
    print(f"train dataset length: {len(dataset['train'])}")

    # add data
    tokenized_input = dataset.map(tokenize_input, batched=True, num_proc=10, remove_columns=["informal", "formal"])
    tokenized_label = dataset.map(tokenize_label, batched=True, num_proc=10, remove_columns=["informal", "formal"])

    max_seq_len = max([len(i) for i in tokenized_input['train']['input_ids']]) + 30 # 30 is for giving some space for the model to generate
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_input['train'], batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    train_label_dataloader = DataLoader(tokenized_label['train'], batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    test_em_dataloader = DataLoader(tokenized_input['test_em'], batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    test_fr_dataloader = DataLoader(tokenized_input['test_fr'], batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    # initialize hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("1 Batch Training...")
    model.train()
    for i, (batch, batch_label) in enumerate(tqdm(zip(train_dataloader, train_label_dataloader))):
        print(f"========================BATCH: {i}=========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
        batch_label = {k: torch.Tensor(v).to(device) for k, v in batch_label.items()}
        optimizer.zero_grad()
        pad_to_add = batch_label['input_ids'].shape[1] - batch['input_ids'].shape[1]
        # add padding to the input
        if pad_to_add > 0:
            batch['input_ids'] = F.pad(batch['input_ids'], (0, pad_to_add), value=tokenizer.pad_token_id)
            batch['attention_mask'] = F.pad(batch['attention_mask'], (0, pad_to_add), value=0)
        lora.mark_only_lora_as_trainable(model)
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch_label['input_ids'])
        loss = outputs.loss
        print(loss.item())
        loss.backward()
        optimizer.step()
            
    torch.save(lora.lora_state_dict(model), opt.checkpoint_path)
        
    model.eval()
    em_pred_list = list()
    em_input_list = list()
    print("Testing... (Entertainment_Music)")    
    for i, (batch) in enumerate(tqdm(zip(test_em_dataloader))):
        print(f"========================BATCH: {i}=========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch[0].items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], 
                                     attention_mask=batch['attention_mask'], 
                                     max_length=max_seq_len, 
                                     generation_config=generation_config)
            input_list = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            pred_list = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        em_pred_list.append(pred_list)
        em_input_list.append(input_list)

        print("##PREDICTION##")
        print(pred_list[0].replace(input_list[0], ""))

    em_pred_list_tmp = [j for i in em_pred_list for j in i]
    em_input_list_tmp = [j for i in em_input_list for j in i]

    refined_pred_list = list()
    for i, p in zip(em_input_list_tmp, em_pred_list_tmp):
        refined_pred_list.append(p.replace(i, ""))
    
    refined_pred_list = [i.replace('\x03', '').replace('\r', '').replace('\x13', '') for i in refined_pred_list]

    # make output dir if not exist
    if opt.output_dir is not None:
        # exist_ok=True: if the directory already exists, do not raise an error
        os.makedirs(opt.output_dir, exist_ok=True)

    with open(opt.output_dir+'/input_em.txt', mode='w') as out:
        for i in em_input_list_tmp:
            out.write(i+'\n')

    with open(opt.output_dir+'/lm_pred_em.txt', mode='w') as out:
        for i in refined_pred_list:
            out.write(i.replace("\n", "")+'\n')

    ##

    fr_pred_list = list()
    fr_input_list = list()
    print("Testing... (Entertainment_Music)")    
    for i, (batch) in enumerate(tqdm(zip(test_fr_dataloader))):
        print(f"========================BATCH: {i}=========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch[0].items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], 
                                     attention_mask=batch['attention_mask'], 
                                     max_length=max_seq_len, 
                                     generation_config=generation_config)
            input_list = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            pred_list = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        fr_input_list.append(input_list)
        fr_pred_list.append(pred_list)

        print("##PREDICTION##")
        print(pred_list[0].replace(input_list[0], ""))
    fr_pred_list_tmp = [j for i in fr_pred_list for j in i]
    fr_input_list_tmp = [j for i in fr_input_list for j in i]

    refined_pred_list = list()
    for i, p in zip(fr_input_list_tmp, fr_pred_list_tmp):
        refined_pred_list.append(p.replace(i, ""))
    
    refined_pred_list = [i.replace('\x03', '').replace('\r', '').replace('\x13', '') for i in refined_pred_list]

    # make output dir if not exist
    if opt.output_dir is not None:
        # exist_ok=True: if the directory already exists, do not raise an error
        os.makedirs(opt.output_dir, exist_ok=True)

    with open(opt.output_dir+'/input_fr.txt', mode='w') as out:
        for i in fr_input_list_tmp:
            out.write(i+'\n')

    with open(opt.output_dir+'/lm_pred_fr.txt', mode='w') as out:
        for i in refined_pred_list:
            out.write(i.replace("\n", "")+'\n')
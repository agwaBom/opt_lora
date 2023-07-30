import torch
from random_seed import set_random_seed
from transformers import OPTModel, OPTConfig, AutoTokenizer, OPTForCausalLM, AutoConfig, AutoModelForCausalLM, DataCollatorWithPadding, AdamW, GPT2Tokenizer, DataCollatorForLanguageModeling, GenerationConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator, infer_auto_device_map
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import argparse
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import os
<<<<<<< HEAD:lora_opt_train_gsm8k.py
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
=======
import math
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
>>>>>>> 7bf95423153fb341b449a42ffdf154dfd85b0001:lora_opt_main.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(123)
import loralib as lora

def tokenize_input(examples):
    type = "gsm8k"
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
    type = "gsm8k"
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
    # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        #model = OPTForCausalLM(config).half()
        # model.model.decoder.layers[47].self_attn.out_proj._parameters['lora_A']
        # model.model.decoder.layers[47].self_attn.out_proj._parameters['lora_B']
        model = AutoModelForCausalLM.from_config(config).half()
    # lora meta device로 되는 것을 돌려놓기
    model.model.decoder.layers[47].self_attn.out_proj.to_empty(device='cuda:1').to(torch.float32)
    # weight init 재설정
    nn.init.kaiming_uniform_(model.model.decoder.layers[47].self_attn.out_proj._parameters['lora_A'], a=math.sqrt(5))
    nn.init.zeros_(model.model.decoder.layers[47].self_attn.out_proj._parameters['lora_B'])

    print(model.model.decoder.layers[47].self_attn.out_proj._parameters['lora_B'].type())
    # "balanced_low_0"
    #device_map = infer_auto_device_map(model)

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
    parser.add_argument("--batch_size", default=30, type=int) # 약 4G 모델
    parser.add_argument("--model", default="facebook/opt-iml-max-1.3b")
    parser.add_argument("--log_dir", default="./runs/opt-iml-max-1.3b")
    #parser.add_argument("-model", default="facebook/opt-125m")
    #parser.add_argument("-log_dir", default="./runs/opt-125m")
    parser.add_argument("--train_data_path", default="./dataset/news_commentary_16_en_de_train.json")
    parser.add_argument("--valid_data_path", default="./dataset/dev.json")

<<<<<<< HEAD:lora_opt_train_gsm8k.py
    #parser.add_argument("--test_data_path_em", default="./GYAFC_Corpus/GYAFC_Corpus/Entertainment_Music/test/test.json")
    #parser.add_argument("--test_data_path_fr", default="./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/test.json")
    #parser.add_argument("--output_dir", default="./lora_gyafc_output/")
    parser.add_argument("--checkpoint_path", default="./lora_models/opt_iml_max_1_3b/r8/gsm8k/opt-iml-1_3b_gsm8k_lora")
    parser.add_argument("--num_epochs", default=10, type=int)
=======
    parser.add_argument("--test_data_path_em", default="./GYAFC_Corpus/GYAFC_Corpus/Entertainment_Music/test/test.json")
    parser.add_argument("--test_data_path_fr", default="./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/test.json")
    parser.add_argument("--output_dir", default="./lora_gyafc_output/")
    parser.add_argument("--checkpoint_path", default="./LoRA/opt-iml-30b_lora_1.pt")
>>>>>>> 7bf95423153fb341b449a42ffdf154dfd85b0001:lora_opt_main.py
    opt =  parser.parse_args()

    print(opt)

    writer = SummaryWriter(opt.log_dir)
    batch_size = opt.batch_size
    model_path = "/home/khyunjin1993/dev/myRepo/knnopt/model_weight/"+opt.model
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
    #try:
    if checkpoint in non_sharded_checkpoint_list:
        model = load_non_sharded_checkpoint(checkpoint, device)
    else:
        model = load_sharded_checkpoint(checkpoint, model_path)
    #except:
    #    print("Loading Local Model Failed. Downloading a New Model ..")
    #    model = OPTForCausalLM.from_pretrained(checkpoint)
    #    os.mkdir('./model_weight/'+checkpoint)
    #    save_pretrained_weight('./model_weight/'+checkpoint, model)
    #    if checkpoint in non_sharded_checkpoint_list:
    #        model = load_non_sharded_checkpoint(checkpoint, device)
    #    else:
    #        model = load_sharded_checkpoint(checkpoint, model_path)

    print("Model Loaded.")
    model.generation_config = generation_config

    # Load Dataset of informal input and formal.ref0 label
    dataset = load_dataset("gsm8k", "main", split="train[:7000]")
    valid_dataset = load_dataset("gsm8k", "main", split="train[7000:]")

    rc_a = "question"
    rc_b = "answer"

    print(f"total dataset length: {len(dataset)}\n")
    print("rc_a and rc_b", rc_a, rc_b)

    # add data
    tokenized_input = dataset.map(tokenize_input, batched=True, num_proc=10, remove_columns=[rc_a, rc_b])
    tokenized_label = dataset.map(tokenize_label, batched=True, num_proc=10, remove_columns=[rc_a, rc_b])
    valid_tokenized_label = valid_dataset.map(tokenize_label, batched=True, num_proc=10, remove_columns=[rc_a, rc_b])

<<<<<<< HEAD:lora_opt_train_gsm8k.py
=======
    max_seq_len = max([len(i) for i in tokenized_input['test_em']['input_ids']]) + 30 # 30 is for giving some space for the model to generate
>>>>>>> 7bf95423153fb341b449a42ffdf154dfd85b0001:lora_opt_main.py
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(tokenized_input, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    train_label_dataloader = DataLoader(tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    valid_label_dataloader = DataLoader(valid_tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    # initialize hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #import IPython; IPython.embed(); exit(1)
    '''
    cnt = 0
    for p in model.parameters():
        if p.requires_grad:
            cnt += 1
            print(cnt, p.name, p.data.shape)
    '''
    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = torch.nn.DataParallel(model)
    for epoch in range(opt.num_epochs):
        total_train_loss = 0
        for i, (batch, batch_label) in enumerate(tqdm(zip(train_dataloader, train_label_dataloader))):
            model.train()
            batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
            batch_label = {k: torch.Tensor(v).to(device) for k, v in batch_label.items()}
            lora.mark_only_lora_as_trainable(model)
            optimizer.zero_grad()
            current_step = epoch * len(train_label_dataloader) + i
            # pad_to_add = batch_label['input_ids'].shape[1] - batch['input_ids'].shape[1]
            # add padding to the input
            # if pad_to_add > 0:
            #     batch['input_ids'] = F.pad(batch['input_ids'], (0, pad_to_add), value=tokenizer.pad_token_id)
            #     batch['attention_mask'] = F.pad(batch['attention_mask'], (0, pad_to_add), value=0)
            
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
                torch.save(lora.lora_state_dict(model.module), opt.checkpoint_path+f"_{current_step}_{round(total_valid_loss/len(valid_label_dataloader), 2)}.pt")
        print("Total Train Loss: ", total_train_loss/len(train_label_dataloader))

    '''    
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
    print("Testing... (Family_Relationships)")    
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
    '''
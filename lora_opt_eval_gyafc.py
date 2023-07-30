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
import math

def tokenize_input(examples):
    type = "gyafc"
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
    type = "gyafc"
    if type == "wmt":
        labels = ["Translate this from English to German: \n" + en + "\nGerman: " + du for en, du in zip(examples['en'], examples['du'])]
    elif type == "gyafc":
        labels = ["Convert the following informal sentence into a formal sentence: \nInformal: " + informal + "\nFormal: " + formal for informal, formal in zip(examples['informal'], examples['formal'])]
    elif type == "gsm8k":
        labels = ["Problem: " + question + "\nSolution: " + answer for question, answer in zip(examples['question'], examples['answer'])]

    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_labels = tokenizer(labels, max_length=500, add_special_tokens=True)
    return model_labels

def load_sharded_checkpoint(checkpoint, path, opt):
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
    model.load_state_dict(torch.load(opt.lora_checkpoint_path, map_location="cuda:2"), strict=False)
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
    parser.add_argument("--batch_size", default=20, type=int) # 약 4G 모델
    parser.add_argument("--model", default="facebook/opt-iml-max-1.3b")
    parser.add_argument("--log_dir", default="./runs/opt-iml-max-1.3b")
    #parser.add_argument("-model", default="facebook/opt-125m")
    #parser.add_argument("-log_dir", default="./runs/opt-125m")
    parser.add_argument("--data_path", default="./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/test.json")

    #parser.add_argument("--test_data_path_em", default="./GYAFC_Corpus/GYAFC_Corpus/Entertainment_Music/test/test.json")
    #parser.add_argument("--test_data_path_fr", default="./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/test.json")
    parser.add_argument("--output_dir", default="./lora_output/opt-iml-1_3b/r8/gyafc_fr/")
    parser.add_argument("--lora_checkpoint_path", default="./lora_models/opt_iml_max_1_3b/r8/gyafc/opt-iml-1_3b_gyafc_em_lora_2100_1.48.pt")
    parser.add_argument("--num_epochs", default=10, type=int)
    opt =  parser.parse_args()

    print(opt)

    writer = SummaryWriter(opt.log_dir)
    batch_size = opt.batch_size
    model_path = "/home/khyunjin1993/dev/myRepo/knnopt/model_weight/" + opt.model
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
    model = load_non_sharded_checkpoint(checkpoint, device)

    '''
    try:
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path, opt)
    except:
        print("Loading Local Model Failed. Downloading a New Model ..")
        model = OPTForCausalLM.from_pretrained(checkpoint)
        os.mkdir('./model_weight/'+checkpoint)
        save_pretrained_weight('./model_weight/'+checkpoint, model)
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)
    '''
    print("Model Loaded.")
    model.generation_config = generation_config

    # Load Dataset of informal input and formal.ref0 label
    if opt.data_path == "gsm8k":
        dataset = load_dataset("gsm8k", "main")
        dataset = dataset["test"]
        rc_a = "question"
        rc_b = "answer"
    else:
        dataset = load_dataset("json", data_files=opt.data_path)
        dataset = dataset["train"]

        if "GYAFC" in opt.data_path:
            rc_a = "informal"
            rc_b = "formal"
        elif "wmt" in opt.data_path:
            rc_a = "du"
            rc_b = "en"

    print(f"total dataset length: {len(dataset)}\n")
    print("rc_a and rc_b", rc_a, rc_b)

    # add data
    tokenized_input = dataset.map(tokenize_input, batched=True, num_proc=10, remove_columns=[rc_a, rc_b])
    tokenized_label = dataset.map(tokenize_label, batched=True, num_proc=10, remove_columns=[rc_a, rc_b])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    test_dataloader = DataLoader(tokenized_input, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    test_label_dataloader = DataLoader(tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    model.load_state_dict(torch.load(opt.lora_checkpoint_path), strict=False)

    model.eval()
    em_pred_list = list()
    em_input_list = list()
    print("Testing...")    
    for i, (batch, batch_label) in enumerate(tqdm(zip(test_dataloader, test_label_dataloader))):
        print(f"========================BATCH/LENGTH: {i}/{len(test_label_dataloader)} =========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
        with torch.no_grad():
            max_seq_len = batch_label['input_ids'].shape[-1] + 10 # 10 is for giving some space for the model to generate
            print("Max Sequence: ",max_seq_len)
            outputs = model.generate(input_ids=batch['input_ids'], 
                                     attention_mask=batch['attention_mask'], 
                                     max_length=256,
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

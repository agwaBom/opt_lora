import torch
import deepspeed
from random_seed import set_random_seed
from transformers import OPTModel, OPTConfig, AutoTokenizer, OPTForCausalLM, AutoConfig, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import tempfile
import gc

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sharded_checkpoint(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config).half()
    model = load_checkpoint_and_dispatch(model, "./model_weight/"+checkpoint, device_map="auto", no_split_module_classes=["OPTDecoderLayer"])
    return model

def save_pretrained_weight(path, model):
    model.save_pretrained(path)
    print(sorted(os.listdir(path)))

if __name__ == "__main__":
    checkpoint = "facebook/opt-66b"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    try:
        model = load_sharded_checkpoint(checkpoint)
    except:
        print("Loading Local Model Failed. Downloading a New Model ..")
        model = OPTForCausalLM.from_pretrained(checkpoint)
        os.mkdir('./model_weight/'+checkpoint)
        save_pretrained_weight('./model_weight/'+checkpoint, model)
        model = load_sharded_checkpoint(checkpoint)
    
    print("Model Loaded.")

    inputs = tokenizer("answer the question below and explain why:\nDo you think doctors' position is only for men?", return_tensors="pt")
    import IPython; IPython.embed(); exit()

    scaler = torch.cuda.amp.GradScaler()

    # Default: 26.3GB
    # half (fp16): 13.6GB
    # model = model.to(device)
    # outputs = model(**inputs.to(device))

    #model.cpu()
    #del model
    #gc.collect()
    #torch.cuda.empty_cache()

    # Default: 39GB
    # half (fp16): 13.6GB
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=False):
        model = model.to(device)
        # Parallel batches of data
        #model = torch.nn.DataParallel(model)
        outputs = model(**inputs.to(device))

        outputs = model.generate(inputs["input_ids"])
    with torch.no_grad():
        model = model.to(device)
        outputs = model(**inputs.to(device))

    #generate_ids = model.module.generate(inputs.input_ids, max_length=100)
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    tokenizer.decode(generate_ids.squeeze())
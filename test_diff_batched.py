from transformers import pipeline
import torch
from datasets import load_dataset, Dataset
import evaluate
from random_seed import set_random_seed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(123)

class MyDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label
    def __len__(self, file):
        return len(file)
    def __getitem__(self, i):
        return {'input': self.input[i], 'label': self.label[i]}

dataset = load_dataset("wmt16", "de-en")
test_data = dataset["test"]["translation"]

model = "facebook/opt-iml-1.3b"
generator = pipeline('text-generation', model=model, max_length = 250, device_map="auto")

input_list = ["</s>Translate German to English:\n" + example['de'] + " => " for example in test_data]
label_list = ["</s>Translate German to English:\n" + example['de'] + " => " + example['en'] for example in test_data]

dataset = MyDataset(input_list, label_list)

import IPython; IPython.embed(); exit()

bleu = evaluate.load("bleu")
pred_list = generator(input_list)
pred_list = [i[0]['generated_text'] for i in pred_list]

total_pred_list = list()
total_label_list = list()

for j, (inp, pred, lab) in enumerate(zip(input_list, pred_list, label_list)):
    replaced_pred = pred.replace(inp, "")
    replaced_label = lab.replace(inp, "")
    pred_list[j] = replaced_pred
    label_list[j] = replaced_label
    total_pred_list.append(replaced_pred)
    total_label_list.append(replaced_label)

result = bleu.compute(predictions=total_pred_list, references=total_label_list)
print(f"Total BLEU: {result}")
try:
    with open('./'+model.replace("/", "_")+'_pred_list.txt', mode='w') as out:
        for pred, lab in zip(total_pred_list, total_label_list):
            out.write("-------------------------------------------------\n")
            out.write("Ground Truth: "+lab+"\nPrediction: "+pred+'\n')
        out.write(f"Total BLEU: {result}")
except:
    import IPython; IPython.embed()
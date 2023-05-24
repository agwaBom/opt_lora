from transformers import pipeline
import torch
from datasets import load_dataset
import evaluate
from random_seed import set_random_seed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(123)

dataset = load_dataset("wmt16", "de-en")
test_data = dataset["test"]["translation"]

model = "facebook/opt-125m"
generator = pipeline('text-generation', model=model, max_length = 250, device_map="auto")

input_list = ["Translate German to English:\n" + example['de'] + " => " for example in test_data]
label_list = ["Translate German to English:\n" + example['de'] + " => " + example['en'] for example in test_data]

print(len(input_list))
bleu = evaluate.load("bleu")

for i in input_list:
    print(generator(i))
    

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
#try:
#    with open('./'+model.replace("/", "_")+'_pred_list_optimal_decode.txt', mode='w') as out:
#        for pred, lab in zip(total_pred_list, total_label_list):
#            out.write("-------------------------------------------------\n")
#            out.write("Ground Truth: "+lab+"\nPrediction: "+pred+'\n')
#        out.write(f"Total BLEU: {result}")
#except:
#    import IPython; IPython.embed()
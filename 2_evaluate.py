import evaluate


if __name__ == "__main__":
    path = './facebook_opt-iml-30b_en_du_pred_list_postprocessing.txt'

    file = open(path, mode='r').read().split('\n')
    bleu = evaluate.load("sacrebleu")
    pred_list = list()
    gt_list = list()
    for line in file:
        if line.startswith('Prediction:'):
            pred_list.append(line.replace("Prediction: ", ""))
        elif line.startswith('Ground Truth:'):
            gt_list.append(line.replace("Ground Truth: ", ""))

    result = bleu.compute(predictions=pred_list, references=gt_list)
    print(result)
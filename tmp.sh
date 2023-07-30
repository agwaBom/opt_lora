python lora_opt_eval_gyafc.py \
    --data_path ./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/test.json \
    --output_dir ./lora_output/opt-iml-30b/r8/gyafc_fr/ \
    --lora_checkpoint_path lora_models/opt_iml_max_30b/r8/gyafc/opt-iml-30b_gyafc_lora_3500_1.39.pt

python lora_opt_eval_gyafc.py \
    --data_path ./GYAFC_Corpus/GYAFC_Corpus/Entertainment_Music/test/test.json \
    --output_dir ./lora_output/opt-iml-30b/r8/gyafc_em/ \
    --lora_checkpoint_path lora_models/opt_iml_max_30b/r8/gyafc/opt-iml-30b_gyafc_lora_3500_1.39.pt

python lora_opt_train_gsm8k_30b.py \
    --checkpoint_path ./lora_models/opt_iml_max_30b/r8/gsm8k/opt-iml-1_3b_gsm8k_lora

python lora_opt_eval_gyafc.py 
python lora_opt_eval_gsm8k.py
python lora_opt_eval_wmt.py

#%%
# Import packages
from transformers import BertTokenizerFast

#%%
# Set arguments
PRETRAIN_MODEL_PATH = "/home/ee303/Documents/agbld/Models/pretrained_models/ckiplab-bert-base-chinese"
OUTPUT_MODEL_NAME = "EComBERT_ruten_adaption_fp16O1"
CHECKPOINT = '300000'

#%%
# Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(PRETRAIN_MODEL_PATH)
tokenizer.save_pretrained(f"./Experiments/{OUTPUT_MODEL_NAME}/checkpoint-{CHECKPOINT}/")

#%%
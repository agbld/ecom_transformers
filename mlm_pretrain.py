#%%
# Import packages
import random
import numpy as np
import torch
import os
from transformers import BertTokenizerFast, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from ruten_utils.RutenItemNamesDataset import RutenItemNamesDataset

#%%
# Set arguments
DB_PATH='/home/ee303/Documents/agbld/Datasets/ruten.db'
PRETRAIN_MODEL_PATH = "/home/ee303/Documents/agbld/Models/pretrained_models/ckiplab-bert-base-chinese"
OUTPUT_MODEL_NAME = "EComBERT_ruten_adaption_fp16O1"

#%%
# Set seed
def setTorchSeed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 2022
setTorchSeed(seed)

#%%
# Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(PRETRAIN_MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(PRETRAIN_MODEL_PATH)

#%%
# Initialize PChome+MOMO dataset
# class CustomDataset(Dataset):
#     def __init__(self, tokenizer):
#         with open("F:/Datasets/PChome/title_desc_corpus_chinese.txt", 'r') as f:
#             self.source = list(f)
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.source)

#     def __getitem__(self, idx):
#         #encoded = self.tokenizer(self.source['train']['text'][idx], return_tensors = 'pt')
#         text = self.source[idx].replace('\n','')
#         # encoded = self.tokenizer(text = text, truncation= True)#, max_length = 128)
#         # return encoded
#         return text
    
# dataset = CustomDataset(tokenizer)

#%%
# sum_txt_len = 0
# max_txt_len = 0
# inspect_len = 100000
# from tqdm import tqdm
# with tqdm(total=inspect_len) as pbar:
#     for i in range(inspect_len):
#         random_idx = random.randint(0, len(dataset)-1)
        
#         sum_txt_len += len(dataset[random_idx]['input_ids'])
#         if len(dataset[random_idx]['input_ids']) > max_txt_len:
#             max_txt_len = len(dataset[random_idx]['input_ids'])
#         # pbar.set_description(f"Processing {random_idx}")
#         pbar.set_postfix({'avg_txt_len': sum_txt_len / (i + 1), 'max_txt_len': max_txt_len})
#         pbar.update(1)
    
#%%
# Initialize Ruten dataset
corpus_dataset = RutenItemNamesDataset(db_path=DB_PATH,
                                table_name='ruten_items',
                                col_item_name='G_NAME',
                                create_db=False,    # set to True to re-create the database
                                verbose=True)

class TokenizedDataset(Dataset):
    def __init__(self, corpus_dataset, tokenizer, corpus_limit = None, maxlength = 128):
        self.corpus_dataset = corpus_dataset
        self.tokenizer = tokenizer
        self.corpus_limit = corpus_limit
        self.maxlength = maxlength

    def __len__(self):
        if self.corpus_limit is not None:
            return self.corpus_limit
        else:
            return len(self.corpus_dataset)

    def __getitem__(self, idx):
        encoded = self.tokenizer(text = self.corpus_dataset[idx], truncation= True, max_length = self.maxlength)
        return encoded
    
dataset = TokenizedDataset(corpus_dataset, tokenizer, maxlength = 128)

# sum_txt_len = 0
# max_txt_len = 0
# inspect_len = 100000
# from tqdm import tqdm
# with tqdm(total=inspect_len) as pbar:
#     for i in range(inspect_len):
#         random_idx = random.randint(0, len(dataset)-1)
        
#         sum_txt_len += len(dataset[random_idx]['input_ids'])
#         if len(dataset[random_idx]['input_ids']) > max_txt_len:
#             max_txt_len = len(dataset[random_idx]['input_ids'])
#         pbar.set_description(f"Processing {random_idx}")
#         pbar.set_postfix({'sum_txt_len': sum_txt_len, 'max_txt_len': max_txt_len})
#         pbar.update(1)

#%%
# Initialize trainer

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=f"./Experiments/{OUTPUT_MODEL_NAME}",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=128, # 64, maxlength = 128 at 15GB
    save_steps=100000,
    save_total_limit=40,
    seed=2022,
    data_seed=2022,
    report_to="wandb",
    logging_steps=500,
    fp16=True,
    fp16_opt_level="O1",
    #resume_from_checkpoint = './RoBERTa_retrained/checkpoint-510000'
    # max_steps = 19578382 # 19578382 / 64 = 305912 steps (1 epoch)
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

#%%
# Train
trainer.train() #resume_from_checkpoint = True)

trainer.save_model(f"./Experiments/{OUTPUT_MODEL_NAME}/checkpoint-latest/")

#%%
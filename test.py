import argparse
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from dataloader import fundchatdataset

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
SENT = "<unused1>"

data_path = 'C:/kogpt2/ChatBotData.csv'
model_path = 'C:/kogpt2/chat_model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default = 10, type = int)
#parser.add_argument("--lr", default = 3e-5, type = float)
parser.add_argument("--batch_size", default = 32, type = int)
parser.add_argument("--warmup_steps", default = 100, type = int)
args = parser.parse_args('')

tokenizer = PreTrainedTokenizerFast.from_pretrained("C:/kogpt2/tokenizer_with_custom_tokens", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained(model_path)

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))

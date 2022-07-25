import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertConfig,BertForTokenClassification
from torch import cuda
from src.data_preprocessing import DataPreprocessing
from src.utils import UtilsForProgram
from src.data_ingestion import DataIngestion





class ModelArchitecture(Dataset):
    def __init__(self,dataframe,tokenizer,max_len,label2id) -> None:
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id=label2id

    def __getitem__(self, index):
        #tokenize and adopt corresponding labels
        sentence = self.data.Sentence[index]
        word_labels = self.data.Word_Labels[index]
        tokenized_sentence, labels = UtilsForProgram.tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        #Add special tokens
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token
        # step 3: truncating/padding
        maxlen = self.max_len
        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask for everything except the padded tokens
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [self.label2id[label] for label in labels]

        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

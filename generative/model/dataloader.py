import json

from icecream import ic
from torch.utils.data import Dataset
from langdetect import detect, DetectorFactory

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import torch

class Dataset1(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, is_mt5):
        fp = open(data_path, 'r')
        self.df = [json.loads(line, strict=False) for line in fp.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        self.languages_map = {
            'bn': 'bn_IN',
            'de': 'de_DE',
            'en': 'en_XX',
            'es': 'es_XX',
            'fr': 'fr_XX',
            'gu': 'gu_IN',
            'hi': 'hi_IN',
            'it': 'it_IT',
            'kn': 'kn_IN',
            'ml': 'ml_IN',
            'mr': 'mr_IN',
            'or': 'or_IN',
            'pa': 'pa_IN',
            'ta': 'ta_IN',
            'te': 'te_IN',
        }
        self.intro_map = {
            'bn': 'ভূমিকা',
            'en': 'Introduction',
            'hi': 'परिचय',
            'kn': 'ಪರಿಚಯ',
            'ml': 'ആമുഖം',
            'mr': 'परिचय',
            'or': 'ପରିଚୟ',
            'pa': 'ਜਾਣ-ਪਛਾਣ',
            'ta': 'அறிமுகம்',
            'te': 'పరిచయం'
        }

        DetectorFactory.seed = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        article = self.df[idx]['article']
        article_title = article['title']
        sections = ""

        # self.septoken = '</s>'

        lang = self.df[idx]['language']
        if lang not in self.languages_map:
            lang='en'
        xlang = lang
        lang = self.languages_map[lang]

        for section in article['sections']:
            if section['title'] == 'Introduction':
                sections = f'{sections} {self.intro_map[xlang]}'
            else:
                sections = f'{sections} {section["title"]}'

        domain = self.df[idx]['domain']
        input_text = f'{lang} {domain} {article_title}'

        target_text = sections
        # self.tokenizer.add_special_tokens({'sep_token': self.septoken})

        input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
        target_encoding = self.tokenizer(lang + target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

        input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
        labels = target_encoding['input_ids']

        if self.is_mt5 == 1:
            labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations


        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'lang': lang, 'domain': domain}

class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

    def setup(self, stage=None):
        # ic("loading train")
        self.train = Dataset1(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
        # ic("loading val")
        self.val = Dataset1(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
        # ic("loading test")
        self.test = Dataset1(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1,shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from transformers import DistilBertTokenizer, BertTokenizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score
from tqdm import tqdm
import numpy as np
from random import sample, seed
import re
import os
import argparse

from pyencoders import VADER
from nltk.tokenize import sent_tokenize

from regressor import style_embedding_evaluation
from extractor import features_array_from_string


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")
BERT_PATH = os.path.join("..","BERT", "bert-base-uncased")

BERT_START_INDEX = 101
BERT_END_INDEX = 102

def set_seed(graine):
    seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)

############# Text Reader ###############
def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)

class BookDataset(Dataset):

    def __init__(self, data_dir, encoder, train, columns, max_len = 512, seed = 1):
        super(BookDataset, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.seed = seed
        self.max_len = max_len
        self.encoder = encoder
        self.columns = columns

        if encoder == "DistilBERT":
          self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
        elif encoder == "BERT":
          self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

        self.authors = sorted([a for a in os.listdir(os.path.join(self.data_dir)) if os.path.isdir(os.path.join(self.data_dir, a))])
        
        self.documents = []
        self.author_documents = []

        print("------------ Reading Corpora ------------", flush=True)
        for author in tqdm(self.authors):
            docs = sorted([doc for doc in os.listdir(os.path.join(self.data_dir, author))])

            for doc in docs:
                content = read(os.path.join(self.data_dir, author, doc))
                self.documents.append(content)
                self.author_documents.append(author)

        self.aut2id = dict(zip(self.authors, list(range(len(self.authors)))))

        if self.train:
            self.documents, _, self.author_documents, _ = train_test_split(self.documents, 
                                                                           self.author_documents,
                                                                           test_size=0.2,
                                                                           stratify=self.author_documents,
                                                                           random_state=self.seed)
        else:
            _, self.documents, _, self.author_documents = train_test_split(self.documents, 
                                                                           self.author_documents,
                                                                           test_size=0.2,
                                                                           stratify=self.author_documents,
                                                                           random_state=self.seed)

        self.nd = len(self.documents)
        self.na = len(self.authors)

    def _process_train_data(self):
        print("------------ Processing Train Corpora ------------", flush=True)
        self.texts = []
        self.features = []
        self.author_chunks = []
        
        for doc, author in tqdm(zip(self.documents, self.author_documents), total = len(self.documents)):
            sentences = sent_tokenize(doc)
            sentences_ids = self.tokenizer(sentences, add_special_tokens=False)['input_ids']

            l=0
            doc_ids = []
            temp = []
            for sentence, sentence_ids in zip(sentences, sentences_ids):
                l+=len(sentence_ids)
                if l>510:
                    if len(temp)>0:
                        self.texts.append(" ".join(temp))
                        self.features.append(features_array_from_string(" ".join(temp), self.columns))
                        self.author_chunks.append(self.aut2id[author])

                        l=len(sentence_ids)
                        doc_ids=sentence_ids
                        temp = [sentence]
                    else:
                        text1, text2 = sentence[:len(sentence)//2], sentence[len(sentence)//2:]
                        self.texts.append(text1)
                        self.features.append(features_array_from_string(text1, self.columns))
                        self.author_chunks.append(self.aut2id[author])
                        self.texts.append(text2)
                        self.features.append(features_array_from_string(text2, self.columns))
                        self.author_chunks.append(self.aut2id[author])
                        l=0
                else:
                   doc_ids += sentence_ids
                   temp += [sentence]

            if l>510:
                if (len(temp)>0)&(len(" ".join(temp))>50):
                    self.texts.append(" ".join(temp))
                    self.features.append(features_array_from_string(" ".join(temp), self.columns))
                    self.author_chunks.append(self.aut2id[author])
                else:
                    text1, text2 = sentence[:len(sentence)//2], sentence[len(sentence)//2:]
                    self.texts.append(text1)
                    self.features.append(features_array_from_string(text1, self.columns))
                    self.author_chunks.append(self.aut2id[author])
                    self.texts.append(text2)
                    self.features.append(features_array_from_string(text2, self.columns))
                    self.author_chunks.append(self.aut2id[author])
            else:
                if (len(temp)>0)&(len(" ".join(temp))>50):
                    self.texts.append(" ".join(temp))
                    self.features.append(features_array_from_string(" ".join(temp), self.columns))
                    self.author_chunks.append(self.aut2id[author])

    def _process_test_data(self):
        print("------------ Processing Test Corpora ------------", flush=True)
        self.texts = []
        
        for doc in tqdm(self.documents, total=len(self.documents)):
            text = []
            sentences = sent_tokenize(doc)
            sentences_ids = self.tokenizer(sentences, add_special_tokens=False)['input_ids']

            l=0
            doc_ids = []
            temp = []
            for sentence, sentence_ids in zip(sentences, sentences_ids):
                l+=len(sentence_ids)
                if l>510:
                    if len(temp)>0:
                        text.append(" ".join(temp))
                        l=len(sentence_ids)
                        doc_ids=sentence_ids
                        temp = [sentence]
                    else:
                        text1, text2 = sentence[:len(sentence)//2], sentence[len(sentence)//2:]
                        text.append(text1)
                        text.append(text2)
                        l=0
                else:
                   doc_ids += sentence_ids
                   temp += [sentence]

            if (len(temp)>0):
                text.append(" ".join(temp))
                self.texts.append(text)

        self.aut_doc_test = np.zeros((self.nd, self.na))
        self.aut_doc_test[[i for i in range(self.nd)],[self.aut2id[author] for author in self.author_documents]] = 1

    def _negative_sample(self, negpairs=10):
        print("------------ Building Pairs ------------", flush=True)

        self.pairs = []
        self.labels = []
        doc_ids = [i for i in range(len(self.texts))]
        for d, a in tqdm(enumerate(self.author_chunks), total=len(doc_ids)):
            self.pairs.append([d, a, d])
            self.labels.append([1,1])

            self.pairs.extend([[d,a,f] for f in sample(doc_ids[:d] + doc_ids[d+1:], negpairs)])
            self.labels.extend([[1,0] for _ in range(negpairs)])

            self.pairs.extend([[d, i, d] for i in np.random.choice(list(range(0,a))+list(range(a+1, self.na)),negpairs)])
            self.labels.extend([[0,1] for _ in range(negpairs)])

            self.pairs.extend([[d, i, f] for i, f in zip(np.random.choice(list(range(0,a))+list(range(a+1, self.na)),negpairs), sample(doc_ids[:d] + doc_ids[d+1:], negpairs))])
            self.labels.extend([[0,0] for _ in range(negpairs)])

    def __getitem__(self, index):

        text_id, author_id, feature_id = self.pairs[index]
        label_author, label_feature = self.labels[index]
        
        result = {'author':author_id,
                  'text':self.texts[text_id],
                  'feature':self.features[feature_id],
                  'label_author':label_author,
                  'label_feature':label_feature
        }

        return result

    def __len__(self):
        return len(self.labels)

    def tokenize_caption(self, caption, device):

        output = self.tokenizer(caption, padding=True, truncation=True, max_length=512, return_tensors='pt')

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        return input_ids.to(device), attention_mask.to(device)

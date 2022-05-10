import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import transformers
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score
from tqdm import tqdm
import numpy as np
import re
import os
import argparse

from regressor import style_embedding_evaluation

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def chunks(*args, **kwargs):
	return list(chunksYielder(*args, **kwargs))
def chunksYielder(l, n):
	"""Yield successive n-sized chunks from l."""
	if l is None:
		return []
	for i in range(0, len(l), n):
		yield l[i:i + n]

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")
BERT_START_INDEX = 101
BERT_END_INDEX = 102

tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH, do_lower_case=True, local_files_only=True)

############# Text Reader ###############
def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)

class book_Dataset(Dataset):
    def __init__(self, books, authors, tokenizer, max_len):
        self.data = books
        self.authors = authors
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        author = self.authors[index]

        doc = self.data[index]
        doc = tokenizer.encode(doc, add_special_tokens=False)

        parts = chunks(doc, (self.max_len -2))
        parts = [[BERT_START_INDEX] + part + [BERT_END_INDEX] for part in parts]

        if (len(parts) > 1) & (len(parts[-1]) < int(self.max_len * 0.3)):
            parts = parts[:-1]

        masks = [[1]* self.max_len for _ in parts]

        masks[-1] = [1]*len(parts[-1]) + [0] * (self.max_len - len(parts[-1]))
        parts[-1] = parts[-1] + [0] * (self.max_len - len(parts[-1]))

        return parts, masks, [author] * len(parts)


class DeepStyle(torch.nn.Module):
    def __init__(self, na):
        super().__init__()
        self.na = na
        self.distilBERT = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        self.drop = torch.nn.Dropout(0.1)
        self.out = torch.nn.Linear(768, na)

    def forward(self, ids, mask):
        distilbert_output = self.distilBERT(ids, mask)
        hidden_state = distilbert_output[0]
        embed = self.drop(hidden_state[:,0])
        output = self.out(embed)
        return output    

    def doc_embedding(self, ids, mask):
        distilbert_output = self.distilBERT(ids, mask)
        hidden_state = distilbert_output[0]
        embed = hidden_state[:,0]
        return torch.mean(embed, dim=0)

    def doc_prediction(self, ids, mask):
        distilbert_output = self.distilBERT(ids, mask)
        hidden_state = torch.mean(distilbert_output[0][:,0], axis=0)
        output = self.out(hidden_state)

        return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type =str,
                        help='Path to dataset directory')
    parser.add_argument('-bs','--batchsize', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--epochs', default=100, type=int,
                        help='Epochs')
    parser.add_argument('-s','--surname', default='', type=str,
                        help='name')
    args = parser.parse_args()

    data_dir = args.dataset
    name=args.surname
    BATCH_SIZE = args.batchsize
    EPOCHS = args.epochs
    
    MAX_LEN = 512
    LEARNING_RATE = 3e-5
    CLIPNORM =  1.0

    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # EPOCHS=5
    # BATCH_SIZE=5
    # name="poutou"

    method = "deep_style_%s" % name

    authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])
    documents = []
    doc2aut = {}
    id_docs = []
    part_mask = []

    for author in tqdm(authors):
        docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, author))])
        id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]

        for doc in docs:
            doc2aut[doc.replace(".txt", "")] = author
            content = read(os.path.join(data_dir, author, doc))
            # tf_content = tf.convert_to_tensor(distilBertEncode(content))
            documents.append(content)

    aut2id = dict(zip(authors, list(range(len(authors)))))
    doc2id = dict(zip(id_docs, list(range(len(id_docs)))))

    nd = len(doc2id)
    na = len(aut2id)

    di2ai = {doc2id[d]: aut2id[a] for d,a in doc2aut.items()}

    doc_train, doc_test, docid_train, docid_test, y_train, y_test = train_test_split(documents, list(doc2id.values()), list(di2ai.values()), test_size=0.2, stratify=list(di2ai.values()))

    print("Build pairs")
    # For testing purpose
    doc_tp = docid_test
    aut_doc_test = np.array(pd.crosstab(di2ai.keys(), di2ai.values()))

    training_set = book_Dataset(doc_train, y_train, tokenizer, MAX_LEN)
    test_set = book_Dataset(doc_test, y_test, tokenizer, MAX_LEN)

    x_test = []
    x_mask_test = []
    for doc, mask, _ in test_set:
        x_test.append(doc)
        x_mask_test.append(mask)

    test_dl = tuple(zip(x_test, x_mask_test, y_test))

    x = []
    x_mask = []
    y = []

    for doc, mask, label in training_set:
        x.extend(doc)
        x_mask.extend(mask)
        y.extend(label)

    train_dl = DataLoader(TensorDataset(torch.LongTensor(x), torch.LongTensor(x_mask), torch.LongTensor(y)), shuffle=True, batch_size = BATCH_SIZE)

    model = DeepStyle(na)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

    def eval_fn(test_dl, aut_doc_test, model):
        
        model.eval()
        with torch.no_grad():
            test_labels = []
            for doc, mask, _ in tqdm(test_dl):

                doc = torch.LongTensor(doc).to(device)
                mask = torch.LongTensor(mask).to(device)

                test_labels.append(model.doc_prediction(doc, mask).cpu().detach().numpy())

            test_labels = np.vstack(test_labels)

            ce = coverage_error(aut_doc_test, test_labels)
            lr = label_ranking_average_precision_score(aut_doc_test, test_labels)*100

            print("coverage, precision")
            print(str(round(ce,2)) + ", "+ str(round(lr,2)))

        return ce, lr

    def fit(epochs, model, loss_fn, opt, train_dl, test_dl, aut_doc_test):

        for epoch in range(1,epochs+1):
            model.train()
            for x, mask, y in tqdm(train_dl):
                x=x.to(device)
                mask=mask.to(device)
                y=y.to(device)

                outputs = model(x, mask)
                loss = loss_fn(outputs, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPNORM)
                opt.step()
                opt.zero_grad()

            ce, lr = eval_fn(test_dl, aut_doc_test, model)

            print("[%d/%d]  F-loss : %.4f, Coverage %.2f, LRAP %.2f" % (epoch, epochs, loss.item(), ce, lr), flush=True)

    print("------------ Beginning Training ------------", flush=True)

    fit(EPOCHS, model, loss_fn, optimizer, train_dl, test_dl, aut_doc_test[doc_tp,:])

    #### Final Evaluation ####
    x = []
    x_mask = []
    y_train = np.array(y_train)

    for doc, mask, _ in training_set:
        x.append(doc)
        x_mask.append(mask)

    doc_embeddings = []
    for doc, mask, _ in zip(x, x_mask, y_train):

        doc = torch.LongTensor(doc).to(device)
        mask = torch.LongTensor(mask).to(device)

        doc_embeddings.append(model.doc_embedding(doc, mask).cpu().detach().numpy())

    doc_embeddings = np.vstack(doc_embeddings)

    author_embeddings = []
    for i in range(na):
        author_embeddings.append(np.mean(doc_embeddings[np.where(y_train==i)[0],:], axis=0))

    author_embeddings = np.vstack(author_embeddings)

    doc_embeddings = []
    for doc, mask, _ in test_dl:

        doc = torch.LongTensor(doc).to(device)
        mask = torch.LongTensor(mask).to(device)

        doc_embeddings.append(model.doc_embedding(doc, mask).cpu().detach().numpy())

    doc_embeddings = np.vstack(doc_embeddings)

    aa = normalize(author_embeddings, axis=1)
    dd = normalize(doc_embeddings, axis=1)
    y_score = normalize( dd @ aa.transpose(),norm="l1")
    ce = coverage_error(aut_doc_test[doc_tp,:], y_score)/na*100
    lr = label_ranking_average_precision_score(aut_doc_test[doc_tp,:], y_score)*100
    print("Final coverage, precision with cosine similarity")
    print(str(round(ce,2)) + ", "+ str(round(lr,2)))

    features = pd.read_csv(os.path.join("data", "gutenberg", "features", "features.csv"), sep=";")

    if not os.path.isdir(os.path.join("results",method)):
        os.mkdir(os.path.join("results",method))

    res_df = style_embedding_evaluation(author_embeddings, features.groupby("author").mean().reset_index(), n_fold=10)
    print(res_df)
    res_df.to_csv(os.path.join("results", "deep_style", "style_%s.csv" % "deep_style"), sep=";")
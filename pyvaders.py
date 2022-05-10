import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from transformers import DistilBertTokenizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score
from tqdm import tqdm
import numpy as np
from random import sample
import re
import os
import argparse

from pyencoders import VADER

from regressor import style_embedding_evaluation
from extractor import features_array_from_string

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

def convert_ids_to_features_array(tokenizer, ids, document, columns):
    c=""
    i=0
    n=len(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)).replace(" ",""))
    while i<n:
        s=document[i]
        c=c+s
        if s==" ":
            n+=1
        i+=1

    return features_array_from_string(c, columns)

class book_Dataset(Dataset):
    def __init__(self, books, authors, tokenizer, max_len, columns, with_features=True):
        self.data = books
        self.authors = authors
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.columns = columns
        self.with_features = with_features

    def __getitem__(self, index):
        author = self.authors[index]

        doc = self.data[index]
        doc_id = self.tokenizer.encode(doc, add_special_tokens=False)

        parts = chunks(doc_id, (self.max_len -2))

        parts = [[BERT_START_INDEX] + part + [BERT_END_INDEX] for part in parts]

        if (len(parts) > 1) & (len(parts[-1]) < int(self.max_len * 0.3)):
            parts = parts[:-1]

        if self.with_features:
            features = [convert_ids_to_features_array(self.tokenizer, part[1:-1], doc, columns) for part in parts]

        masks = [[1]* self.max_len for _ in parts]

        masks[-1] = [1]*len(parts[-1]) + [0] * (self.max_len - len(parts[-1]))
        parts[-1] = parts[-1] + [0] * (self.max_len - len(parts[-1]))

        if self.with_features:
            return parts, masks, [author] * len(parts), features
        else:
            return parts, masks, [author] * len(parts)

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
    parser.add_argument('-n','--negpairs', default=1, type=int,
                        help='Number of negative pairs to sample')
    args = parser.parse_args()

    data_dir = args.dataset
    name=args.surname
    BATCH_SIZE = args.batchsize
    EPOCHS = args.epochs
    NEGPAIRS = args.negpairs
    
    MAX_LEN = 512
    LEARNING_RATE = 3e-5
    CLIPNORM =  1.0

    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # EPOCHS=5
    # BATCH_SIZE=5
    # NEGPAIRS=5
    # name="poutou"

    method = "pyvaders_%s" % name

    authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])[:10]
    documents = []
    doc2aut = {}
    id_docs = []
    part_mask = []

    print("------------ Reading Corpora ------------", flush=True)
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

    print("%d authors and %d documents.\n" % (na, nd), flush=True)

    di2ai = {doc2id[d]: aut2id[a] for d,a in doc2aut.items()}

    doc_train, doc_test, docid_train, docid_test, y_train, y_test = train_test_split(documents, list(doc2id.values()), list(di2ai.values()), test_size=0.2, stratify=list(di2ai.values()))

    # For testing purpose
    doc_tp = docid_test
    aut_doc_test = np.array(pd.crosstab(di2ai.keys(), di2ai.values()))

    features = pd.read_csv(os.path.join("data", "gutenberg", "features", "features.csv"), sep=";").sort_values(by=["author", "id"])

    features_test = features.drop(["id", "author", 'needn\'t', 'couldn\'t', 'hasn\'t', 'mightn\'t', 'you\'ve', 'shan\'t', 'aren',
        'weren\'t', 'mustn', 'shan', 'should\'ve', 'mightn', 'needn', 'hadn\'t',
        'aren\'t', 'hadn', 'that\'ll', '£', '€', '<', '\'', '^', '~'], axis=1)

    columns = features_test.columns

    features_test = features_test.to_numpy()
    
    stdScale = StandardScaler()
    stdScale.fit(features_test)

    features_test = features_test[doc_tp,:]

    training_set = book_Dataset(doc_train, y_train, tokenizer, MAX_LEN, columns)
    test_set = book_Dataset(doc_test, y_test, tokenizer, MAX_LEN, columns, with_features=False)

    x_test = []
    x_mask_test = []

    print("------------ Tokenizing Test Set ------------", flush=True)
    for doc, mask, _ in tqdm(test_set, total=len(doc_test)):
        x_test.append(doc)
        x_mask_test.append(mask)

    test_dl = tuple(zip(x_test, x_mask_test, y_test))

    x = []
    x_mask = []
    x_features = []
    y = []

    print("------------ Tokenizing Train Set ------------", flush=True)
    for doc, mask, label, x_f in tqdm(training_set, total=len(doc_train)):
        x.extend(doc)
        x_mask.extend(mask)
        x_features.extend(x_f)
        y.extend(label)

    x_features = stdScale.transform(np.vstack(x_features))
    y = np.array(y)

    print("%d total samples with an average of %.2f samples per document.\n" % (len(x_mask), len(x_mask)/nd), flush=True)

    print("------------ Building Pairs ------------", flush=True)
    data_pairs = []
    features_train = []
    labels = []
    doc_ids = [i for i in range(len(x))]
    for d, a in tqdm(enumerate(y), total=len(y)):
        data_pairs.append([d, a, d])
        labels.append([1,1])

        data_pairs.extend([[d,a, f] for f in sample(doc_ids[:d] + doc_ids[d+1:], NEGPAIRS)])
        labels.extend([[1,0] for _ in range(NEGPAIRS)])

        data_pairs.extend([[d, i, d] for i in np.random.choice(y[y!=a],NEGPAIRS)])
        labels.extend([[0,1] for _ in range(NEGPAIRS)])

        data_pairs.extend([[d, i, f] for i, f in zip(np.random.choice(y[y!=a],NEGPAIRS), sample(doc_ids[:d] + doc_ids[d+1:], NEGPAIRS))])
        labels.extend([[0,0] for _ in range(NEGPAIRS)])

    print("%d training pairs created.\n" % len(labels), flush=True)

    train_dl = DataLoader(TensorDataset(torch.LongTensor(data_pairs),
                                        torch.tensor(labels, dtype=torch.float32)),
                                        shuffle=True, batch_size = BATCH_SIZE)

    x = torch.LongTensor(x)
    x_mask = torch.LongTensor(x_mask)
    x_features = torch.tensor(x_features, dtype=torch.float32)

    model = VADER(na, 300, L=10)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

    def eval_fn(test_dl, aut_doc_test, model, features):
        
        model.eval()
        with torch.no_grad():

            doc_embeddings = []
            aut_vars = []
            aut_embeddings = []

            for doc, mask, labels in tqdm(test_dl):

                doc = torch.LongTensor(doc).to(device)
                mask = torch.LongTensor(mask).to(device)

                doc_emb, _ = model(doc, mask)
                doc_embeddings.append(doc_emb.cpu().detach().numpy())

            ll = [i for i in range(model.na)]
            for i in range(0, model.na, BATCH_SIZE):

                ids = torch.LongTensor(ll[i:i+BATCH_SIZE]).to(device)
                aut_embeddings.append(model.mean_author(ids).cpu().detach().numpy())
                aut_vars.append(model.logvar_author(ids).cpu().detach().numpy())

        doc_embeddings = np.vstack(doc_embeddings)
        aut_embeddings = np.vstack(aut_embeddings)
        aut_vars = np.vstack(aut_vars)

        aa = normalize(aut_embeddings, axis=1)
        dd = normalize(doc_embeddings, axis=1)
        y_score = normalize( dd @ aa.transpose(),norm="l1")
        ce = coverage_error(aut_doc_test, y_score)/na*100
        lr = label_ranking_average_precision_score(aut_doc_test, y_score)*100
        print("coverage, precision", flush=True)
        print(str(round(ce,2)) + ", "+ str(round(lr,2)))

        res_df = style_embedding_evaluation(aut_embeddings, features.groupby("author").mean().reset_index(), n_fold=10)
        print(res_df)

        np.save(os.path.join("results", method, "aut_%s.npy" % method), aut_embeddings)
        np.save(os.path.join("results", method, "aut_var_%s.npy" % method), aut_vars)
        np.save(os.path.join("results", method, "doc_%s.npy" % method), doc_embeddings)
        res_df.to_csv(os.path.join("results", method, "style_%s.csv" % method), sep=";")
    
        return ce, lr

    def fit(epochs, model, loss_fn, opt, train_dl, x, x_mask, x_features, test_dl, aut_doc_test, features):

        for epoch in range(1,epochs+1):
            model.train()
            opt.zero_grad()
            for x_train, y_train in tqdm(train_dl):
                
                doc , author, doc_f = torch.tensor_split(x_train, 3, dim=1)
                
                mask = x_mask[doc.squeeze()]
                doc = x[doc.squeeze()]
                doc_f = x_features[doc_f.squeeze()]

                doc = doc.to(device)
                mask = mask.to(device)
                doc_f = doc_f.to(device)
                author = author.to(device)
                y_train= y_train.to(device)

                loss, f_loss, a_loss, p_loss = model.loss_VIB(author, doc, mask, doc_f, y_train, loss_fn)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPNORM)
                opt.step()

            ce, lr = eval_fn(test_dl, aut_doc_test, model, features)

            print("[%d/%d]  F-loss : %.4f, A-loss : %.4f, I-loss : %.4f, Coverage %.2f, LRAP %.2f" % (epoch, epochs, f_loss, a_loss, p_loss, ce, lr), flush=True)

    print("------------ Beginning Training ------------", flush=True)

    if not os.path.isdir(os.path.join("results",method)):
        os.mkdir(os.path.join("results",method))

    fit(EPOCHS, model, criterion, optimizer, train_dl, x, x_mask, x_features, test_dl, aut_doc_test[doc_tp,:], features)
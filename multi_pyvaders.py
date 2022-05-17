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
from random import sample, seed
import re
import os
import argparse

from pyencoders import VADER

from regressor import style_embedding_evaluation
from extractor import features_array_from_string

import idr_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend = 'nccl',
                        init_method = 'env://',
                        world_size=idr_torch.size,
                        rank = idr_torch.rank)

torch.cuda.set_device(idr_torch.local_rank)
device = torch.device("cuda")

# Setting up the device for GPU usage

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

def convert_ids_to_features_array(tokenizer, start, ids, document, columns):
    c=""
    i=0
    n=len(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)).replace(" ",""))
    count=0
    for s in document:
        c = c+s
        if s!=" ":
            count+=1
        if count==n:
            break

    return features_array_from_string(c, columns), len(c)+start

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

        start=0
        features = []
        if self.with_features:
            for part in parts:
                feature, n = convert_ids_to_features_array(self.tokenizer, start, part[1:-1], doc, self.columns)
                start=n

                features.append(feature)

        masks = [[1]* self.max_len for _ in parts]

        masks[-1] = [1]*len(parts[-1]) + [0] * (self.max_len - len(parts[-1]))
        parts[-1] = parts[-1] + [0] * (self.max_len - len(parts[-1]))

        if self.with_features:
            return parts, masks, [author] * len(parts), features
        else:
            return parts, masks, [author] * len(parts)

if __name__ == "__main__":

    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']

    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes, master node is ", MASTER_ADDR)
    print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))


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
    parser.add_argument('-lr','--learningrate', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('-t','--test', default=0, type=int,
                        help='test')
    args = parser.parse_args()

    data_dir = args.dataset
    dataset = data_dir.split(os.sep)[-1]
    name=args.surname
    BATCH_SIZE = args.batchsize
    batch_size_per_gpu = BATCH_SIZE // idr_torch.size
    EPOCHS = args.epochs
    NEGPAIRS = args.negpairs
    LEARNING_RATE = args.learningrate
    test = args.test
    
    MAX_LEN = 512
    CLIPNORM =  1.0

    # LEARNING_RATE = 5e-5
    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # EPOCHS=5
    # BATCH_SIZE=5
    # NEGPAIRS=5
    # name="poutou"

    method = "multi_vaders_%s" % name

    set_seed(13)
    print("Random init is : %f" % torch.rand(1), flush=True)

    authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])
    if test==1:
        authors = authors[:10]

    documents = []
    doc2aut = {}
    id_docs = []
    part_mask = []

    if idr_torch.rank == 0:
        print("------------ Reading Corpora ------------", flush=True)
   
    for author in authors:
        docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, author))])
        id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]

        for doc in docs:
            doc2aut[doc.replace(".txt", "")] = author
            content = read(os.path.join(data_dir, author, doc))
            documents.append(content)

    aut2id = dict(zip(authors, list(range(len(authors)))))
    doc2id = dict(zip(id_docs, list(range(len(id_docs)))))

    nd = len(doc2id)
    na = len(aut2id)

    print("%d authors and %d documents.\n" % (na, nd), flush=True)

    di2ai = {doc2id[d]: aut2id[a] for d,a in doc2aut.items()}

    doc_train, doc_test, docid_train, docid_test, y_train, y_test = train_test_split(documents, list(doc2id.values()), list(di2ai.values()), test_size=0.2, stratify=list(di2ai.values()), random_state=13)

    # For testing purpose
    doc_tp = docid_test
    aut_doc_test = np.array(pd.crosstab(di2ai.keys(), di2ai.values()))

    features = pd.read_csv(os.path.join("data", dataset, "features", "features.csv"), sep=";").sort_values(by=["author", "id"])

    # features = features[features.author.isin(authors)]

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

    if idr_torch.rank == 0:
        print("------------ Tokenizing Test Set ------------", flush=True)
    
    for doc, mask, _ in test_set:
        x_test.append(doc)
        x_mask_test.append(mask)

    test_dl = tuple(zip(x_test, x_mask_test, y_test))

    x = []
    x_mask = []
    x_features = []
    y = []

    if idr_torch.rank == 0:
        print("------------ Tokenizing Train Set ------------", flush=True)
    
    for doc, mask, label, x_f in training_set:
        x.extend(doc)
        x_mask.extend(mask)
        x_features.extend(x_f)
        y.extend(label)

    x_features = stdScale.transform(np.vstack(x_features))
    y = np.array(y)

    print("%d total samples with an average of %.2f samples per document.\n" % (len(x_mask), len(x_mask)/nd), flush=True)

    if idr_torch.rank == 0:
        print("------------ Building Pairs ------------", flush=True)
    
    data_pairs = []
    features_train = []
    labels = []
    doc_ids = [i for i in range(len(x))]
    for d, a in enumerate(y):
        data_pairs.append([d, a, d])
        labels.append([1,1])

        data_pairs.extend([[d,a, f] for f in sample(doc_ids[:d] + doc_ids[d+1:], NEGPAIRS)])
        labels.extend([[1,0] for _ in range(NEGPAIRS)])

        data_pairs.extend([[d, i, d] for i in np.random.choice(y[y!=a],NEGPAIRS)])
        labels.extend([[0,1] for _ in range(NEGPAIRS)])

        data_pairs.extend([[d, i, f] for i, f in zip(np.random.choice(y[y!=a],NEGPAIRS), sample(doc_ids[:d] + doc_ids[d+1:], NEGPAIRS))])
        labels.extend([[0,0] for _ in range(NEGPAIRS)])

    print("%d training pairs created.\n" % len(labels), flush=True)

    train_dataset = TensorDataset(torch.LongTensor(data_pairs),
                                  torch.tensor(labels, dtype=torch.float32))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                   num_replicas = idr_torch.size,
                                                                   rank = idr_torch.rank,
                                                                   shuffle=True)

    train_dl = DataLoader(train_dataset,
                          batch_size = batch_size_per_gpu,
                          shuffle = False,
                          num_workers = 0,
                          pin_memory = True,
                          sampler = train_sampler)

    x = torch.LongTensor(x)
    x_mask = torch.LongTensor(x_mask)
    x_features = torch.tensor(x_features, dtype=torch.float32)

    model = VADER(na, 300, L=10)
    model.to(device)

    ddp_model = DDP(model, device_ids=[idr_torch.local_rank])

    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params = ddp_model.parameters(), lr = LEARNING_RATE)

    def eval_fn(test_dl, aut_doc_test, model, features, style=True):
        
        model.eval()
        with torch.no_grad():

            doc_embeddings = []
            aut_vars = []
            aut_embeddings = []

            for doc, mask, labels in test_dl:

                doc = torch.LongTensor(doc).to(device)
                mask = torch.LongTensor(mask).to(device)

                doc_emb, _ = model(doc, mask)
                doc_embeddings.append(doc_emb.mean(dim=0).cpu().detach().numpy())

            ll = [i for i in range(model.na)]
            for i in range(0, model.na, batch_size_per_gpu):

                ids = torch.LongTensor(ll[i:i+batch_size_per_gpu]).to(device)
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

        if style:
            res_df = style_embedding_evaluation(aut_embeddings, features.groupby("author").mean().reset_index().sort_values("author"), n_fold=10)
            print(res_df)
            res_df.to_csv(os.path.join("results", method, "style_%s.csv" % method), sep=";")
    
        np.save(os.path.join("results", method, "aut_%s.npy" % method), aut_embeddings)
        np.save(os.path.join("results", method, "aut_var_%s.npy" % method), aut_vars)
        np.save(os.path.join("results", method, "doc_%s.npy" % method), doc_embeddings)
        
        return ce, lr

    def fit(epochs, model, loss_fn, opt, train_dl, x, x_mask, x_features, test_dl, aut_doc_test, features):

        if idr_torch.rank == 0:
            ce, lr = eval_fn(test_dl, aut_doc_test, model, features)

        for epoch in range(1,epochs+1):
            model.train()
            opt.zero_grad()

            for x_train, y_train in train_dl:
                
                doc , author, doc_f = torch.tensor_split(x_train, 3, dim=1)
                
                mask = x_mask[doc.squeeze()]
                doc = x[doc.squeeze()]
                doc_f = x_features[doc_f.squeeze()]

                doc = doc.to(device, non_blocking = True)
                mask = mask.to(device, non_blocking = True)
                doc_f = doc_f.to(device, non_blocking = True)
                author = author.to(device, non_blocking = True)
                y_train= y_train.to(device, non_blocking = True)

                loss, f_loss, a_loss, p_loss = model.loss_VIB(author, doc, mask, doc_f, y_train, loss_fn)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPNORM)
                opt.step()

            if (idr_torch.rank == 0):
                ce, lr = eval_fn(test_dl, aut_doc_test, model, features, style=(epoch%5 == 0))
                print("[%d/%d]  F-loss : %.4f, A-loss : %.4f, I-loss : %.4f, Coverage %.2f, LRAP %.2f" % (epoch, epochs, f_loss, a_loss, p_loss, ce, lr), flush=True)

    if idr_torch.rank == 0:
        print("------------ Beginning Training ------------", flush=True)

        if not os.path.isdir(os.path.join("results",method)):
            os.mkdir(os.path.join("results",method))

    fit(EPOCHS, model, criterion, optimizer, train_dl, x, x_mask, x_features, test_dl, aut_doc_test[doc_tp,:], features)
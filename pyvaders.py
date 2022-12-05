import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score
from tqdm import tqdm
import numpy as np
from random import sample, seed
import os
import argparse

from pyencoders import VADER
from datasets import BookDataset

from regressor import style_embedding_evaluation
from extractor import features_array_from_string

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def set_seed(graine):
    seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)

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
    parser.add_argument('-lr','--learningrate', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('-t','--test', default=0, type=int,
                        help='test')
    args = parser.parse_args()

    data_dir = args.dataset
    dataset = data_dir.split(os.sep)[-1]
    name=args.surname
    BATCH_SIZE = args.batchsize
    EPOCHS = args.epochs
    NEGPAIRS = args.negpairs
    LEARNING_RATE = args.learningrate
    test = args.test
    
    # MAX_LEN = 512
    # CLIPNORM =  1.0

    # LEARNING_RATE = 5e-5
    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # EPOCHS=5
    # BATCH_SIZE=5
    # NEGPAIRS=5
    # name="poutou"
    # dataset = data_dir.split(os.sep)[-1]

    method = "pyvaders_%s" % name

    set_seed(13)
    print("Random init is : %f" % torch.rand(1), flush=True)

    features = pd.read_csv(os.path.join("data", dataset, "features", "features.csv"), sep=";").sort_values(by=["author", "id"])

    features = features.drop(["id", "author", 'needn\'t', 'couldn\'t', 'hasn\'t', 'mightn\'t', 'you\'ve', 'shan\'t', 'aren',
        'weren\'t', 'mustn', 'shan', 'should\'ve', 'mightn', 'needn', 'hadn\'t',
        'aren\'t', 'hadn', 'that\'ll', '£', '€', '<', '\'', '^', '~'], axis=1)

    columns = features.columns

    features = features.to_numpy()
    
    stdScale = StandardScaler()
    stdScale.fit(features)

    dataset_train = BookDataset(data_dir, encoder = "DistilBERT", train=True, columns = columns, max_len = 512, seed = 13)
    dataset_train._process_train_data()

    dataset_train.features = stdScale.transform(dataset_train.features)

    dataset_train._negative_sample(negpairs=NEGPAIRS)

    dataset_test = BookDataset(data_dir, encoder = "DistilBERT", train=False, columns = columns, max_len = 512, seed = 13)
    dataset_test._process_test_data()

    train_dl = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = VADER(dataset_train.na, 300, "DistilBERT", L=10)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

    def eval_fn(test_dataset, model, features, style=True):
        
        model.eval()
        with torch.no_grad():

            doc_embeddings = []
            aut_vars = []
            aut_embeddings = []

            for text in tqdm(test_dataset.texts):

                input_ids, attention_masks = test_dataset.tokenize_caption(text, device)

                doc_emb, _ = model(input_ids, attention_masks)
                doc_embeddings.append(doc_emb.mean(dim=0).cpu().detach().numpy())

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
        ce = coverage_error(test_dataset.aut_doc_test, y_score)/test_dataset.na*100
        lr = label_ranking_average_precision_score(test_dataset.aut_doc_test, y_score)*100
        print("coverage, precision", flush=True)
        print(str(round(ce,2)) + ", "+ str(round(lr,2)))

        if style:
            res_df = style_embedding_evaluation(aut_embeddings, features.groupby("author").mean().reset_index(), n_fold=10)
            print(res_df)
            res_df.to_csv(os.path.join("results", method, "style_%s.csv" % method), sep=";")

        np.save(os.path.join("results", method, "aut_%s.npy" % method), aut_embeddings)
        np.save(os.path.join("results", method, "aut_var_%s.npy" % method), aut_vars)
        np.save(os.path.join("results", method, "doc_%s.npy" % method), doc_embeddings)
        
        return ce, lr


    def fit(epochs, model, loss_fn, opt, train_dl, test_dataset, features):

        ce, lr = eval_fn(test_dataset, model, features)

        for epoch in range(1,epochs+1):
            model.train()
            
            for batch in tqdm(train_dl):
                
                author, doc, doc_f, y_a, y_f = batch.values()

                input_ids, attention_masks = dataset_train.tokenize_caption(doc, device)

                doc_f = doc_f.to(device)
                author = author.to(device)
                y_a = y_a.to(device)
                y_f = y_f.to(device)

                loss, f_loss, a_loss, p_loss = model.loss_VIB(author, input_ids, attention_masks, doc_f, y_a, y_f, loss_fn)

                opt.zero_grad()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPNORM)
                opt.step()

            ce, lr = eval_fn(test_dataset, model, features, style=(epoch%5 == 0))

            print("[%d/%d]  F-loss : %.4f, A-loss : %.4f, I-loss : %.4f, Coverage %.2f, LRAP %.2f" % (epoch, epochs, f_loss, a_loss, p_loss, ce, lr), flush=True)

    print("------------ Beginning Training ------------", flush=True)

    if not os.path.isdir(os.path.join("results",method)):
        os.mkdir(os.path.join("results",method))

    fit(EPOCHS, model, criterion, optimizer, train_dl, dataset_test, features)
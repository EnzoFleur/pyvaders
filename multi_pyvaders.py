import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score
from tqdm import tqdm
import numpy as np
from random import sample, seed
import re
import os
import argparse
from datetime import datetime

from transformers import get_linear_schedule_with_warmup

from pyencoders import VADER
from datasets import BookDataset

from regressor import style_embedding_evaluation
from extractor import features_array_from_string

import idr_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Setting up the device for GPU usage
dist.init_process_group(backend = 'nccl',
                        init_method = 'env://',
                        world_size=idr_torch.size,
                        rank = idr_torch.rank)

torch.cuda.set_device(idr_torch.local_rank)
device = torch.device("cuda")

def set_seed(graine):
    seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)

ft_dict = {"True":True, "False":False}

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
    parser.add_argument('-e','--encoder', default="DistilBERT", type=str,
                        help='Text encoder')
    parser.add_argument('-a','--alpha', default=1/2, type=float,
                        help='Alpha parameter value')
    parser.add_argument('-wa','--attention', default="False", type=str,
                        help='Alpha parameter value')
    parser.add_argument('-f','--freeze', default="False", type=str,
                        help='Alpha parameter value')
    args = parser.parse_args()

    data_dir = args.dataset
    dataset = data_dir.split(os.sep)[-1]
    name=args.surname
    BATCH_SIZE = args.batchsize
    batch_size_per_gpu = BATCH_SIZE // idr_torch.size
    EPOCHS = args.epochs
    NEGPAIRS = args.negpairs
    LEARNING_RATE = args.learningrate
    ENCODER = args.encoder
    ALPHA = args.alpha
    ATTENTION = ft_dict[args.attention]
    FREEZE = ft_dict[args.freeze]

    MAX_LEN = 512
    # CLIPNORM =  1.0

    # LEARNING_RATE = 5e-5
    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # EPOCHS=5
    # BATCH_SIZE=5
    # NEGPAIRS=5
    # name="poutou"
    # dataset = data_dir.split(os.sep)[-1]

    method = "multi_vaders_%s" % name

    set_seed(13)
    print("Random init is : %f" % torch.rand(1), flush=True)

    features = pd.read_csv(os.path.join("data", dataset, "features", "features.csv"), sep=";").sort_values(by=["author", "id"])

    features = features.drop(["id", 'needn\'t', 'couldn\'t', 'hasn\'t', 'mightn\'t', 'you\'ve', 'shan\'t', 'aren',
        'weren\'t', 'mustn', 'shan', 'should\'ve', 'mightn', 'needn', 'hadn\'t',
        'aren\'t', 'hadn', 'that\'ll', '£', '€', '<', '\'', '^', '~'], axis=1)

    columns = features.drop(["author"], axis=1).columns
    
    dataset_train = BookDataset(data_dir, encoder = "DistilBERT", train=True, columns = columns, max_len = 512, seed = 42)

    # features = features[features.author.isin(dataset_train.authors)]

    stdScale = StandardScaler()
    stdScale.fit(features.drop(["author"], axis=1))

    dataset_train._process_train_data()

    dataset_train.features = stdScale.transform(dataset_train.features)

    dataset_train._negative_sample(negpairs=NEGPAIRS)

    dataset_test = BookDataset(data_dir, encoder = "DistilBERT", train=False, columns = columns, max_len = 512, seed = 42)
    dataset_test._process_test_data()

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                   num_replicas = idr_torch.size,
                                                                   rank = idr_torch.rank,
                                                                   shuffle=True)

    train_dl = DataLoader(dataset_train,
                          batch_size = batch_size_per_gpu,
                          shuffle = False,
                          num_workers = 0,
                          pin_memory = True,
                          sampler = train_sampler)

    model = VADER(na=dataset_train.na, doc_r=300, encoder=ENCODER, L=10, alpha=ALPHA, with_attention=ATTENTION, finetune=FREEZE)
    model.to(device)

    ddp_model = DDP(model, device_ids=[idr_torch.local_rank])

    criterion = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.Adam(params = ddp_model.parameters(), lr = LEARNING_RATE)

    optimizer = torch.optim.Adam(params = [{'params': ddp_model.module.params.encoder.parameters(), 'lr': 5e-4,
                                           'params': ddp_model.module.params.classifier.parameters(), 'lr': LEARNING_RATE}])

    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

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

            ll = [i for i in range(model.module.na)]
            for i in range(0, model.module.na, BATCH_SIZE):

                ids = torch.LongTensor(ll[i:i+BATCH_SIZE]).to(device)
                aut_embeddings.append(model.module.mean_author(ids).cpu().detach().numpy())
                aut_vars.append(model.module.logvar_author(ids).cpu().detach().numpy())

        doc_embeddings = np.vstack(doc_embeddings)
        aut_embeddings = np.vstack(aut_embeddings)
        aut_vars = np.vstack(aut_vars)

        aa = normalize(aut_embeddings, axis=1)
        dd = normalize(doc_embeddings, axis=1)
        y_score = normalize( dd @ aa.transpose(),norm="l1")
        ce = coverage_error(test_dataset.aut_doc_test, y_score)/test_dataset.na*100
        lr = label_ranking_average_precision_score(test_dataset.aut_doc_test, y_score)*100

        if style:
            res_df = style_embedding_evaluation(aut_embeddings, features.groupby("author").mean().reset_index().sort_values(by=["author"]), n_fold=10)
            print(res_df)
            res_df.to_csv(os.path.join("results", method, "style_%s.csv" % method), sep=";")

        np.save(os.path.join("results", method, "aut_%s.npy" % method), aut_embeddings)
        np.save(os.path.join("results", method, "aut_var_%s.npy" % method), aut_vars)
        np.save(os.path.join("results", method, "doc_%s.npy" % method), doc_embeddings)
        
        return ce, lr

    def fit(epochs, model, loss_fn, optimizer, scheduler, train_dl, test_dataset, features):

        if idr_torch.rank == 0:
            ce, lr = eval_fn(test_dataset, model, features)

        for epoch in range(1,epochs+1):

            if (epoch > 4) & FREEZE:
                for param in model.module.encoder.parameters():
                    param.requires_grad=False

            model.train()
            
            if idr_torch.rank == 0: start = datetime.now()

            for batch in train_dl:
                
                author, doc, doc_f, y_a, y_f = batch.values()

                input_ids, attention_masks = dataset_train.tokenize_caption(doc, device)

                doc_f = doc_f.to(device)
                author = author.to(device)
                y_a = y_a.to(device)
                y_f = y_f.to(device)

                loss, f_loss, a_loss, p_loss = model.module.loss_VIB(author, input_ids, attention_masks, doc_f, y_a, y_f, loss_fn)

                optimizer.zero_grad()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPNORM)
                optimizer.step()

                scheduler.step()

            if (idr_torch.rank == 0):
                
                if (epoch % 5 == 0):
                    ce, lr = eval_fn(test_dataset, model, features, style=True)

                print("[%d/%d] in %s F-loss : %.4f, A-loss : %.4f, I-loss : %.4f, Coverage %.2f, LRAP %.2f" % (epoch, epochs, str(datetime.now() - start), f_loss, a_loss, p_loss, ce, lr), flush=True)


    if (idr_torch.rank == 0):
        print("------------ Beginning Training ------------", flush=True)

        if not os.path.isdir(os.path.join("results",method)):
            os.mkdir(os.path.join("results",method))

    fit(EPOCHS, ddp_model, criterion, optimizer, scheduler, train_dl, dataset_test, features)
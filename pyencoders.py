import torch
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")

class MLP(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()

            self.bn = torch.nn.BatchNorm1d(input_size)
            self.do = torch.nn.Dropout(p=0.1)
            self.fc1 = torch.nn.Linear(input_size, input_size)
            self.tanh = torch.nn.Tanh()
            self.fc2 = torch.nn.Linear(input_size, output_size)
        
        def forward(self, x):
            x = self.fc1(self.do(x))
            x = self.tanh(x)
            x = self.bn(x)
            x = self.fc2(self.do(x))
            return x

class VADER(torch.nn.Module):
    def __init__(self, na, doc_r, beta=1e-12, alpha=1/2, L=1):
        super().__init__()
        self.na = na
        self.beta = beta
        self.L = L
        self.doc_r = doc_r
        self.alpha = alpha

        self.drop = torch.nn.Dropout(0.1)

        self.encoder = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        self.a_authors = torch.nn.Parameter(torch.rand(1))
        self.b_authors = torch.nn.Parameter(torch.rand(1))

        self.a_features = torch.nn.Parameter(torch.rand(1))
        self.b_features = torch.nn.Parameter(torch.rand(1))

        self.doc_mean = MLP(768, self.doc_r)
        self.doc_var = MLP(768, self.doc_r)

        self.mean_author = torch.nn.Embedding(self.na, self.doc_r)
        torch.nn.init.normal_(self.mean_author.weight, mean=0.0, std=1.0)
        
        self.logvar_author = torch.nn.Embedding(self.na, self.doc_r)
        torch.nn.init.uniform_(self.logvar_author.weight, a=-0.5, b=0.5)

    def reparameterize(self, mean, logvar):
        
        eps = torch.normal(mean=0.0, std=1.0, size=mean.shape).to(device)

        return eps * torch.sqrt(torch.exp(logvar)) + mean

    def logistic_classifier_features(self, features, doc_emb, apply_sigmoid=True):

        distance = torch.sqrt(torch.sum(torch.square(features - doc_emb), dim = 1))

        logits = -torch.exp(self.a_features) * distance + self.b_features

        if apply_sigmoid:
            logits = torch.sigmoid(logits)

        return logits

    def logistic_classifier(self, x, apply_sigmoid=True):

        logits = -torch.exp(self.a_authors) * x + self.b_authors

        if apply_sigmoid:
            logits = torch.sigmoid(logits)

        return logits

    def forward(self, ids, mask):
        
        distilbert_output = self.encoder(ids, mask)
        hidden_state = distilbert_output[0]
        doc_emb = hidden_state[:,0]
        
        doc_mean = self.doc_mean(doc_emb)
        doc_var = self.doc_var(doc_emb)

        return torch.mean(doc_mean, dim=0), torch.mean(doc_var, dim=0)

    def loss_VIB(self, authors, doc, mask, features, y, criterion):

        y_authors, y_features = torch.tensor_split(y, 2, dim=1)
        y_authors = y_authors.squeeze()
        y_features = y_features.squeeze()

        distilbert_output = self.encoder(doc, mask)
        hidden_state = distilbert_output[0]
        doc_emb = self.drop(hidden_state[:,0])

        doc_mean = self.doc_mean(doc_emb)
        doc_var = self.doc_var(doc_emb)

        author_mean = self.mean_author(authors).squeeze()
        author_logvar = self.logvar_author(authors).squeeze()

        author_loss = 0
        feature_loss = 0
        prior_loss = 0

        for _ in range(self.L):

            doc_emb = self.reparameterize(doc_mean, doc_var)
            aut_emb = self.reparameterize(author_mean, author_logvar)

            features_probs = self.logistic_classifier_features(features, doc_emb)

            distance = torch.sqrt(torch.sum(torch.square(doc_emb - aut_emb), dim=1))

            author_probs = self.logistic_classifier(distance)

            author_loss += criterion(author_probs, y_authors)
            feature_loss += criterion(features_probs, y_features)

        prior_loss += 0.5 * torch.sum(torch.square(author_mean) + torch.exp(author_logvar) - author_logvar -1)
        prior_loss += 0.5 * torch.sum(torch.square(doc_mean) + torch.exp(doc_var) - doc_var -1)

        author_loss *= (1-self.alpha)/self.L
        feature_loss *= self.alpha/self.L
        prior_loss *= self.beta

        loss = feature_loss + author_loss + prior_loss

        return loss, feature_loss.item(), author_loss.item(), prior_loss.item()

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

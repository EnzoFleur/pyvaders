import torch
import torch.nn as nn
from transformers import DistilBertModel, BertModel
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")
BERT_PATH = os.path.join("..", "BERT", "bert-base_uncased")

class DAN(nn.Module):

    def __init__(self, input_dim, hidden, r):
        super(DAN, self).__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.r = r

        self.do1 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.do2 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, r)

    def forward(self, x):

        x = x.mean(dim=1)
        x = self.do1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.do2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x


class MLP(nn.Module):
        def __init__(self, input_size, output_size):
            super(MLP, self).__init__()

            self.input_size = input_size
            self.output_size = output_size

            self.mlp = nn.Sequential(*[
                nn.Dropout(0.2),
                nn.Linear(self.input_size, self.input_size),
                nn.BatchNorm1d(self.input_size),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(self.input_size, self.output_size),
                nn.BatchNorm1d(self.output_size),
            ])
        
        def forward(self, x):

            x = self.mlp(x)

            return x

class VADER(nn.Module):
    def __init__(self, na, doc_r, encoder, beta=1e-12, alpha=1/2, L=1):
        super(VADER, self).__init__()
        self.na = na
        self.beta = beta
        self.L = L
        self.doc_r = doc_r
        self.alpha = alpha

        self.drop = nn.Dropout(0.1)

        if encoder=="DistilBERT":
            self.encoder = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        elif encoder=="BERT":
            self.encoder = BertModel.from_pretrained(BERT_PATH)

        self.a_authors = nn.Parameter(torch.rand(1))
        self.b_authors = nn.Parameter(torch.rand(1))

        self.a_features = nn.Parameter(torch.rand(1))
        self.b_features = nn.Parameter(torch.rand(1))

        self.doc_mean = MLP(768, self.doc_r)
        self.doc_var = MLP(768, self.doc_r)

        self.mean_author = nn.Embedding(self.na, self.doc_r)
        nn.init.normal_(self.mean_author.weight, mean=0.0, std=1.0)
        
        self.logvar_author = nn.Embedding(self.na, self.doc_r)
        nn.init.uniform_(self.logvar_author.weight, a=-0.5, b=0.5)

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

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        
        return masked_outputs

    def forward(self, ids, mask):
        
        encoder_output = self.encoder(input_ids=ids, attention_mask=mask)
        hidden_state = encoder_output[0]

        hidden_state = self.compute_masked_means(hidden_state, mask)
        
        doc_mean = self.doc_mean(hidden_state)
        doc_var = self.doc_var(hidden_state)

        return doc_mean, doc_var

    def loss_VIB(self, authors, ids, mask, features, y_authors, y_features, criterion):

        doc_mean, doc_var = self(ids, mask)

        author_mean = self.mean_author(authors)
        author_logvar = self.logvar_author(authors)

        author_loss = 0
        feature_loss = 0
        prior_loss = 0

        for _ in range(self.L):

            doc_emb = self.reparameterize(doc_mean, doc_var)
            aut_emb = self.reparameterize(author_mean, author_logvar)

            features_probs = self.logistic_classifier_features(features, doc_emb, apply_sigmoid=False)

            distance = torch.sqrt(torch.sum(torch.square(doc_emb - aut_emb), dim=1))

            author_probs = self.logistic_classifier(distance, apply_sigmoid=False)

            author_loss += criterion(author_probs, y_authors.float())
            feature_loss += criterion(features_probs, y_features.float())

        prior_loss += 0.5 * torch.sum(torch.square(author_mean) + torch.exp(author_logvar) - author_logvar -1)
        prior_loss += 0.5 * torch.sum(torch.square(doc_mean) + torch.exp(doc_var) - doc_var -1)

        author_loss *= (1-self.alpha)/self.L
        feature_loss *= self.alpha/self.L
        prior_loss *= self.beta

        loss = feature_loss + author_loss + prior_loss

        return loss, feature_loss.item(), author_loss.item(), prior_loss.item()

class DeepStyle(nn.Module):
    def __init__(self, na):
        super(DeepStyle, self).__init__()
        self.na = na
        self.distilBERT = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(768, na)

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

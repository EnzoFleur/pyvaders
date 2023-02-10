import torch
import torch.nn as nn
from transformers import DistilBertModel, BertModel
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from itertools import chain
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")
BERT_PATH = os.path.join("..", "BERT", "bert-base-uncased")

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

            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(self.input_size, self.input_size)
            self.bn1 = nn.BatchNorm1d(self.input_size)
            self.fc2 = nn.Linear(self.input_size, self.output_size)
            self.bn2 = nn.BatchNorm1d(self.output_size)
        
        def forward(self, x):

            x = self.dropout(x)
            x = self.fc1(x)
            x = F.tanh(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.bn2(self.fc2(x))

            return x

class VADER(nn.Module):
    def __init__(self, na, doc_r, encoder, beta=1e-12, alpha=1/2, L=1, with_attention=False, finetune=True):
        super(VADER, self).__init__()
        self.na = na
        self.beta = beta
        self.L = L
        self.doc_r = doc_r
        self.alpha = alpha
        self.with_attention = with_attention
        self.finetune = finetune

        self.drop = nn.Dropout(0.2)

        if encoder=="DistilBERT":
            self.encoder = DistilBertModel.from_pretrained(DISTILBERT_PATH, output_hidden_states=True)
            self.num_hidden_layers = 7

        elif encoder=="BERT":
            self.encoder = BertModel.from_pretrained(BERT_PATH, output_hidden_states=True)
            self.num_hidden_layers = 13

        # for param in self.encoder.parameters():
        #     param.requires_grad = self.finetune

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

        self.layer_weights = nn.Parameter(torch.tensor([1] * self.num_hidden_layers, dtype=torch.float))

        self.params = nn.ParameterList([self.a_authors, self.b_authors, self.a_features, self.b_features,
                                            self.layer_weights])

        if self.with_attention:
            self.q = nn.Parameter(torch.normal(0.0, 0.1, size=(1,768)))
            self.w_h = nn.Parameter(torch.normal(0.0, 0.1, size=(768,768)))

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

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

    def compute_attention_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)

        masked_outputs = self.attention(masked_outputs)/partition
        
        return masked_outputs

    def forward(self, ids, mask):
        
        encoder_output = self.encoder(input_ids=ids, attention_mask=mask)
        hidden_state = torch.stack(encoder_output["hidden_states"])

        weight_factor = self.layer_weights.view(self.num_hidden_layers, 1, 1, 1).expand(hidden_state.shape)
        hidden_state = (weight_factor * hidden_state).sum(dim=0) / self.layer_weights.sum()

        if self.with_attention:
            hidden_state = self.compute_attention_masked_means(hidden_state, mask)
        else:
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

# ##### TESTING #####

# class AttentionPooling(nn.Module):
#     def __init__(self, num_layers, hidden_size, hiddendim_fc):
#         super(AttentionPooling, self).__init__()
#         self.num_hidden_layers = num_layers
#         self.hidden_size = hidden_size
#         self.hiddendim_fc = hiddendim_fc
#         self.dropout = nn.Dropout(0.1)

#         q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
#         self.q = nn.Parameter(torch.from_numpy(q_t)).float()
#         w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
#         self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

#     def forward(self, all_hidden_states):
#         hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
#                                      for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
#         hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
#         out = self.attention(hidden_states)
#         out = self.dropout(out)
#         return out

#     def attention(self, h):
#         v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
#         v = F.softmax(v, -1)
#         v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
#         v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
#         return v

# from transformers import DistilBertTokenizer

# tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)

# sentences = ["She doesn’t study German on Monday.",
#              "Does she live in Paris?",
#              "He doesn’t teach math.",
#              "Cats hate water.",
#              "Every child likes an ice cream.",
#              "My brother takes out the trash.",
#              "The course starts next Sunday.",
#              "She swims every morning.",
#              "I don’t wash the dishes.",
#              "We see them every week."]

# tokens = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
# input_ids = tokens['input_ids']
# attention_masks = tokens['attention_mask']
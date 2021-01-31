import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from ..utils import sparse_mx_to_torch_sparse_tensor
from .model import Model
from .dataset import load


def train(model, adj, diff, features, sample_size=2000, save_to_pickle='model.pkl', verbose=False):

    nb_epochs = 3000
    patience = 30
    lr = 0.001
    l2_coef = 0.0

    ft_size = features.shape[1]

    batch_size = 4

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        lbl = lbl.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
        
		ba = torch.FloatTensor(ba)
		bd = torch.FloatTensor(bd)
        bf = torch.FloatTensor(bf)
		
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()

        model.train()
        optimiser.zero_grad()

        logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        loss.backward()
        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_to_pickle)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
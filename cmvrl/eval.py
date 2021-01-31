import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from .model import Model
from .dataset import load


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret
		
def evaluate_node_classification(model, dataset_name, adj, diff, features, labels, idx_train, idx_val, idx_test):
        
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis]) 
    
    embeds, _ = model.embed(features, adj, diff, sparse, None)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]
    
    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]    
    
    xent = nn.CrossEntropyLoss()
    accs = []
    wd = 0.01 if dataset_name == 'citeseer' else 0.0
    
    hid_units = model.hidden_dim    
    nb_classes = np.unique(labels).shape[0]

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        #log.cuda()
        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)
    print(accs.mean().item(), accs.std().item())
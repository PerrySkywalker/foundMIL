import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        x = x.unsqueeze(0)
        ## x: N x L
        x = x.squeeze(0)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights( A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class ABMIL(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(ABMIL, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
    def forward(self, **kwargs): ## x: N x L
        h = kwargs['data'].float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        x = h.squeeze(0)
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        logits = self.classifier(afeat) ## K x num_cls
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict


if __name__ == "__main__":
    data = torch.randn((1, 6000, 512)).cuda()
    model = ABMIL().cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)

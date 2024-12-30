import torch.nn as nn
import torch


class ABMIL(nn.Module):
    def __init__(self, n_classes = 2, in_dim=512, hidden_dim=512, dropout=0.3, is_norm=True, *args, **kwargs):
        super().__init__()
        self._fc1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid()
        )
        self.attention_weight = nn.Linear(hidden_dim // 2, 1)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.is_norm = is_norm
    def forward(self, x):
        x = self._fc1(x)
        A_U = self.attention_U(x)
        A_V = self.attention_V(x)
        A = self.attention_weight(A_U * A_V).transpose(1, 2)
        if self.is_norm:
            A = torch.softmax(A, dim=-1)
        x = torch.bmm(A, x).squeeze(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
if __name__ == '__main__':
    model = ABMIL()
    data = torch.rand((1, 1000, 512))
    out = model(data)
    print(out)
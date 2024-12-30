
# python3 main.py --project=$PROJECT_NAME --datasets=tcga --tcga_sub=brca \
# --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=5 \
# --model=rrtmil --pool=attn --n_trans_layers=2 --da_act=tanh --title=brca_plip_rrtmil \
# --all_shortcut --crmsa_k=1 --input_dim=512 --seed=2021
from models.rrt_modules import rrt
import torch.nn as nn
import torch
model_params = {
    'input_dim': 512,
    'n_classes': 2,
    'dropout': 0.25,
    'act': 'relu',
    'region_num': 8,
    'pos': 'none',
    'pos_pos': 0,
    'pool': 'attn',
    'peg_k': 7,
    'drop_path': 0.,
    'n_layers': 2,
    'n_heads': 8,
    'attn': 'rmsa',
    'da_act': 'tanh',
    'trans_dropout': 0.1,
    'ffn': False,
    'mlp_ratio': 4.0,
    'trans_dim': 64,
    'epeg': True,
    'min_region_num': 0.0,
    'qkv_bias': True,
    'epeg_k': 15,
    'epeg_2d': False,
    'epeg_bias': True,
    'epeg_type': 'attn',
    'region_attn': 'native',
    'peg_1d': False,
    'cr_msa': True,
    'crmsa_k': 1,
    'all_shortcut': True,
    'crmsa_mlp': False,
    'crmsa_heads': 8,
    }
class RRT(nn.Module):
    def __init__(self, n_classes = 2, in_dim=512, hidden_dim=512, *args, **kwargs):
        super().__init__()
        model_params['input_dim'] = hidden_dim
        model_params['n_classes'] = n_classes
        self._fc1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.rrt = rrt.RRTMIL(**model_params)
    def forward(self, x):
        x = self._fc1(x)
        logits = self.rrt(x)
        return logits
if __name__ == '__main__':
    model = RRT()
    x = torch.randn((1,1000,512))
    y = model(x)
    print(y)
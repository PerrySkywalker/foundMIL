import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR 
from models import *
from wsi_dataset import WsiDataset
from torch.utils.data import DataLoader
from utils.utils import *
from topk.svm import SmoothTop1SVM
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="模型训练脚本")

    # 超参数
    parser.add_argument('--n_classes', type=int, default=2, help="输出类别数量")
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size 大小")
    parser.add_argument('--log_dir', type=str, default='logs', help="日志文件保存路径")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--max_epochs', type=int, default=100, help="最多训练多少个 epoch")
    parser.add_argument('--patience', type=int, default=20, help="在多少个 epoch 后 AUC 没有增长时停止训练")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help="训练设备")
    parser.add_argument('--model_name', type=str, default='CLAM_MB', help="模型名称")
    # 
    parser.add_argument('--instance_eval', type=bool, default=False, help="only for clam")
    parser.add_argument('--in_dim', type=int, default=512, help="输入特征的维度")
    parser.add_argument('--hidden_dim', type=int, default=512, help="隐藏层的特征的维度")
    # cr, svm
    parser.add_argument('--inst_loss', type=str, default='cr', help="clam_loss")
    parser.add_argument('--n_fold', type=int, default=0, help="交叉验证的 fold 编号")
    parser.add_argument('--data_dir', type=str, default='/home/perry/nvme3n1/TCGA-Brain/All_Features/CONCH/All_Features/', help="数据目录")
    parser.add_argument('--csv_dir', type=str, default='csv/task1/', help="CSV 文件夹路径")
    
    # 解析参数
    args = parser.parse_args()
    return args

def main():
    # 解析参数
    args = parse_args()

    # 固定随机种子
    set_random_seed(args.seed)

    # 日志文件路径
    log_dir = f"{args.log_dir}/{args.model_name}/fold{args.n_fold}/"
    # CSV文件路径
    csv_dir = f"{args.csv_dir}/fold{args.n_fold}.csv"
    # 损失函数
    if 'CLAM' in args.model_name and args.inst_loss == 'svm':
        criterion = SmoothTop1SVM(n_classes = 2).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss()

    # 定义模型
    model = create_model('models', args.model_name, args.n_classes, args.in_dim, args.hidden_dim, instance_eval=args.instance_eval)

    # 模型参数初始化
    model.apply(init_weights_he)
    # 模型加载到设备
    model.to(device=args.device)

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 定义学习率衰减
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # 数据集
    train_dataset = WsiDataset(data_dir=args.data_dir, csv_path=csv_dir, state='train')
    val_dataset = WsiDataset(data_dir=args.data_dir, csv_path=csv_dir, state='val')

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 模型训练
    if args.model_name == 'CLAM_MB' and args.instance_eval:
        model = train_model_clam(
        model=model,
        device=args.device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir
    )
    else:
        model = train_model(
            model=model,
            device=args.device,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.max_epochs,
            patience=args.patience,
            log_dir=log_dir
        )

if __name__ == '__main__':
    main()
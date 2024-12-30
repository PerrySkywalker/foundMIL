import argparse
from utils.utils import *
from models import ABMIL
from wsi_dataset.wsi_dataset import WsiDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="模型测试脚本")
    parser.add_argument('--n_fold', type=int, default=0, help="交叉验证的 fold 编号")
    parser.add_argument('--data_dir', type=str, default='/home/perry/nvme0n1/WMK/TCGA-READ-COAD/UNI', help="数据目录")
    parser.add_argument('--csv_dir', type=str, default='/home/perry/Desktop/TestCode/Liver/TCGA_COAD_READ/', help="CSV 文件夹路径")
    parser.add_argument('--log_dir', type=str, default='logs/', help="交叉验证的 fold 编号")
    parser.add_argument('--model_name', type=str, default='WiKG', help="模型名称")
    parser.add_argument('--in_dim', type=int, default=1024, help="输入特征的维度")
    parser.add_argument('--n_classes', type=int, default=2, help="输出类别数量")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help="训练设备")
    # 解析参数
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    model = create_model('models', args.model_name, args.n_classes, args.in_dim)
    model_params = f'{args.log_dir}/{args.model_name}/fold{args.n_fold}'
    ckpt_files = glob.glob(os.path.join(model_params, '*.ckpt'))
    if len(ckpt_files) == 0 or len(ckpt_files) > 1:
        raise('存在多个模型参数')
    print(f'load {ckpt_files[0]}')
    model.load_state_dict(torch.load(ckpt_files[0]))
    csv_dir = f"{args.csv_dir}/fold{args.n_fold}.csv"
    test_dataset = WsiDataset(data_dir=args.data_dir, csv_path=csv_dir, state='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    model.eval()
    device = args.device
    test_epoch_outputs = []
    df_test = pd.DataFrame()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for features, labels in tqdm(test_loader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(features)
        loss = criterion(logits, labels)
        test_epoch_outputs.append({'logits':logits.detach(), 'labels':labels.detach(), 'losses':loss.detach()})
    
    df_test = compute_metrics_and_log(test_epoch_outputs, df_test, 0)
    print('test_result')
    print(df_test)
    df_test.to_csv(f'{model_params}/test_log.csv', index=False)
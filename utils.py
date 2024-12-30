import copy
import glob
import os

from tqdm import tqdm
from models import ABMIL
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, f1_score, cohen_kappa_score
import random
import numpy as np
import torch
import torch.nn as nn

def create_model(module_name, class_name, *args, **kwargs):
    # 动态导入模块
    module = __import__(module_name, fromlist=[class_name])
    
    # 获取类对象
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    
    return instance

def compute_metrics_and_log(epoch_outputs, df, cur_epoch):
    """
    计算分类任务的准确率、AUC、F1 分数和 loss，并将结果记录到 CSV 文件中。

    :param train_epoch_outputs: 包含模型输出的列表，每个元素是一个字典，包含 'logits', 'labels', 'loss'
    :param df: 用于存储结果的 Pandas DataFrame
    :param cur_epoch: 当前的 epoch 编号
    :return: 更新后的 DataFrame
    """
    # 将 logits、labels 和 loss 从 train_epoch_outputs 中提取并拼接
    labels = torch.cat([x['labels'] for x in epoch_outputs], dim=0).cpu().numpy()
    logits = torch.softmax(torch.cat([x['logits'] for x in epoch_outputs], dim=0), dim=-1).cpu().numpy()
    losses = torch.stack([x['losses'] for x in epoch_outputs]).mean().item()  # 计算平均 loss
    predictions = np.argmax(logits, axis=1)  # 假设 logits 是未经过 sigmoid 的值
    # 检查 logits 的形状
    if len(logits.shape) == 2:  # 二分类问题
        auroc = roc_auc_score(labels, logits[:, 1])  # 计算 AUROC
        auprc = average_precision_score(labels, logits[:, 1]) # 计算 AUPRC
    else:  # 多分类问题
        auroc = roc_auc_score(labels, logits, multi_class='ovr')  # 多分类 AUC
        auprc = average_precision_score(labels, logits, multi_class='ovr')
    # 计算准确率和 F1 分数
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')  # 使用加权平均计算 F1 分数
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    kappa_score = cohen_kappa_score(labels, predictions)
    # 将结果添加到 DataFrame 中
    new_row = {
        'epoch': cur_epoch,
        'loss': losses,
        'acc': accuracy,
        'auroc': auroc,
        'auprc': auprc,
        'precision': precision,
        'recall': recall,
        'kappa_score': kappa_score,
        'f1': f1
    }
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    return df
def set_random_seed(seed=42):
    """
    固定 Python 中所有常见的随机种子，确保代码的可复现性。

    :param seed: 随机种子值，默认为 42
    """
    # 固定 Python 内置的 random 模块的种子
    random.seed(seed)

    # 固定 NumPy 的随机种子
    np.random.seed(seed)

    # 固定 PyTorch 的随机种子（如果安装了 PyTorch）
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except NameError:
        print("PyTorch 未安装，跳过设置 PyTorch 种子。")
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已固定为: {seed}")
def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    patience=5,
    log_dir = 'logs'
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    epochs_no_improve = 0
    try:
        # 检查目录是否已经存在
        if os.path.exists(log_dir):
            # 如果目录存在，检查是否为空
            if os.listdir(log_dir):
                raise ValueError(f"目录 '{log_dir}' 已存在且不为空，无法创建。请检查 log_dir。")
            else:
                print(f"目录 '{log_dir}' 已存在，但为空。")
        else:
            # 如果目录不存在，递归创建目录
            os.makedirs(log_dir)
            print(f"目录 '{log_dir}' 创建成功。")
    except Exception as e:
        # 捕获其他可能的异常
        raise RuntimeError(f"创建目录 '{log_dir}' 时发生错误: {e}")
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    train_epoch_outputs = []
    val_epoch_outputs = []
    for epoch in range(num_epochs):
        model.train()
        print(f'train epoch: {epoch}')
        for features, labels in tqdm(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_epoch_outputs.append({'logits':logits.detach(), 'labels':labels.detach(), 'losses':loss.detach()})
        df_train = compute_metrics_and_log(train_epoch_outputs, df_train, epoch)
        df_train.to_csv(f'{log_dir}/train_log.csv', index=False)
        train_epoch_outputs.clear()
        # 获取表头和最后一行
        headers = df_train.columns
        last_row = df_train.iloc[-1].round(4)
        print('-----------train_log-----------')
        # 确定每列的最大宽度
        col_widths = [max(len(str(headers[i])), len(str(last_row[i]))) for i in range(len(headers))]

        # 打印表头（居中对齐）
        header_str = "".join([str(headers[i]).center(col_widths[i] + 2) for i in range(len(headers))])
        print(header_str)

        # 打印数据（居中对齐）
        row_str = "".join([str(last_row[i]).center(col_widths[i] + 2) for i in range(len(last_row))])
        print(row_str)
        scheduler.step()
        model.eval()
        for features, labels in tqdm(val_loader):
            features = features.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                logits = model(features)
            loss = criterion(logits, labels)
            val_epoch_outputs.append({'logits':logits.detach(), 'labels':labels.detach(), 'losses':loss.detach()})
        df_val = compute_metrics_and_log(val_epoch_outputs, df_val, epoch)
        df_val.to_csv(f'{log_dir}/val_log.csv', index=False)
        headers = df_val.columns
        last_row = df_val.iloc[-1].round(4)
        print('-----------val_log-----------')
        # 确定每列的最大宽度
        col_widths = [max(len(str(headers[i])), len(str(last_row[i]))) for i in range(len(headers))]

        # 打印表头（居中对齐）
        header_str = "".join([str(headers[i]).center(col_widths[i] + 2) for i in range(len(headers))])
        print(header_str)

        # 打印数据（居中对齐）
        row_str = "".join([str(last_row[i]).center(col_widths[i] + 2) for i in range(len(last_row))])
        print(row_str)
        val_epoch_outputs.clear()
        val_auc = df_val['auroc'].iloc[-1]
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_files = glob.glob(os.path.join(log_dir, '*.ckpt'))
            torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}-best_auc_{best_auc:.4f}_best.ckpt'))
            for file_path in ckpt_files:
                os.remove(file_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping!")
            break
    model.load_state_dict(best_model_wts)
    return model







def train_model_clam(
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    patience=5,
    log_dir = 'logs'
):
    
    bag_weight = 0.8
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    epochs_no_improve = 0
    try:
        # 检查目录是否已经存在
        if os.path.exists(log_dir):
            # 如果目录存在，检查是否为空
            if os.listdir(log_dir):
                raise ValueError(f"目录 '{log_dir}' 已存在且不为空，无法创建。请检查 log_dir。")
            else:
                print(f"目录 '{log_dir}' 已存在，但为空。")
        else:
            # 如果目录不存在，递归创建目录
            os.makedirs(log_dir)
            print(f"目录 '{log_dir}' 创建成功。")
    except Exception as e:
        # 捕获其他可能的异常
        raise RuntimeError(f"创建目录 '{log_dir}' 时发生错误: {e}")
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    train_epoch_outputs = []
    val_epoch_outputs = []
    for epoch in range(num_epochs):
        model.train()
        print(f'train epoch: {epoch}')
        train_inst_loss = 0.
        for features, labels in tqdm(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, Y_prob, Y_hat, _, instance_dict = model(features, labels)
            instance_loss = instance_dict['instance_loss']
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value
            bag_loss = criterion(logits, labels)
            loss = bag_weight * bag_loss + (1-bag_weight) * instance_loss
            loss.backward()
            optimizer.step()
            train_epoch_outputs.append({'logits':logits.detach(), 'labels':labels.detach(), 'losses':loss.detach()})
        df_train = compute_metrics_and_log(train_epoch_outputs, df_train, epoch)
        df_train.to_csv(f'{log_dir}/train_log.csv', index=False)
        train_epoch_outputs.clear()
        # 获取表头和最后一行
        headers = df_train.columns
        last_row = df_train.iloc[-1].round(4)
        print('-----------train_log-----------')
        # 确定每列的最大宽度
        col_widths = [max(len(str(headers[i])), len(str(last_row[i]))) for i in range(len(headers))]

        # 打印表头（居中对齐）
        header_str = "".join([str(headers[i]).center(col_widths[i] + 2) for i in range(len(headers))])
        print(header_str)

        # 打印数据（居中对齐）
        row_str = "".join([str(last_row[i]).center(col_widths[i] + 2) for i in range(len(last_row))])
        print(row_str)
        scheduler.step()
        model.eval()
        for features, labels in tqdm(val_loader):
            features = features.to(device)
            labels = labels.to(device)
            # only bag loss
            with torch.no_grad():
                logits, Y_prob, Y_hat, _, instance_dict = model(features, labels)
            loss = criterion(logits, labels)
            val_epoch_outputs.append({'logits':logits.detach(), 'labels':labels.detach(), 'losses':loss.detach()})
        df_val = compute_metrics_and_log(val_epoch_outputs, df_val, epoch)
        df_val.to_csv(f'{log_dir}/val_log.csv', index=False)
        headers = df_val.columns
        last_row = df_val.iloc[-1].round(4)
        print('-----------val_log-----------')
        # 确定每列的最大宽度
        col_widths = [max(len(str(headers[i])), len(str(last_row[i]))) for i in range(len(headers))]

        # 打印表头（居中对齐）
        header_str = "".join([str(headers[i]).center(col_widths[i] + 2) for i in range(len(headers))])
        print(header_str)

        # 打印数据（居中对齐）
        row_str = "".join([str(last_row[i]).center(col_widths[i] + 2) for i in range(len(last_row))])
        print(row_str)
        val_epoch_outputs.clear()
        val_auc = df_val['auroc'].iloc[-1]
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_files = glob.glob(os.path.join(log_dir, '*.ckpt'))
            torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}-best_auc_{best_auc:.4f}_best.ckpt'))
            for file_path in ckpt_files:
                os.remove(file_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping!")
            break
    model.load_state_dict(best_model_wts)
    return model
if __name__ == '__main__':
    model = create_model('models', 'ABMIL')
    a = 1
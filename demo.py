import pandas as pd
import os
import numpy as np
import scipy.stats as stats
def auc_acc(path):
    auc = []
    acc = []
    for _ in os.listdir(path):
        if _ == 'fold1':
            continue
        values = pd.read_csv(f"{path}/{_}/result.csv").values[0]
        acc.append(values[1])
        auc.append(values[6])
    mean_auc = np.mean(auc)
    std_auc = np.std(auc, ddof=1)  # 使用样本标准差
    auc_se = std_auc / np.sqrt(len(auc))
    confidence_interval_auc = stats.t.interval(0.95, len(auc)-1, loc=mean_auc, scale=auc_se)
    mean_acc = np.mean(acc)
    std_acc = np.std(acc, ddof=1)  # 使用样本标准差
    acc_se = std_acc / np.sqrt(len(auc))
    confidence_interval_acc = stats.t.interval(0.95, len(auc)-1, loc=mean_auc, scale=acc_se)
    # 打印结果
    print(f"5个AUC值的平均值为: {mean_auc}")
    print(f"5个AUC值的标准差为: {std_auc}")
    print(f"5个AUC值的95%置信区间为: {confidence_interval_auc}")
    # 打印结果
    print(f"5个ACC值的平均值为: {mean_acc}")
    print(f"5个ACC值的标准差为: {std_acc}")
    print(f"5个ACC值的95%置信区间为: {confidence_interval_acc}")


if __name__ == '__main__':
    auc_acc('logs_zhiliao/config/ABMIL/')
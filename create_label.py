import random
import pandas as pd
import os
from collections import defaultdict

from sklearn.model_selection import KFold

def create_fold(data_label:dict, nfold=5, val_ratio=0.2, save_dir='create_csv/fold_io'):
    data = list(data_label.keys())
    nfold = KFold(n_splits=nfold, shuffle=True)
    for i, (train, test) in enumerate(nfold.split(data)):
        random.shuffle(train)
        random.shuffle(train)
        train_len = len(train)
        val_len = int(train_len * val_ratio)
        train_len = train_len - val_len
        val = train[train_len:]
        val = [data[_] for _ in val]
        train = train[:train_len]
        train = [data[_] for _ in train]
        test = [data[_] for _ in test]
        train_label = [int(float(data_label[_])) for _ in train]
        val_label = [int(float(data_label[_])) for _ in val]
        test_label = [int(float(data_label[_])) for _ in test]
        df1 = pd.DataFrame({'train':train, 'train_label':train_label}).astype(str)
        df2 = pd.DataFrame({'val':val, 'val_label':val_label}).astype(str)
        df3 = pd.DataFrame({'test':test, 'test_label':test_label}).astype(str)
        df = pd.concat([df1, df2, df3], axis=1)
        df.to_csv(f'{save_dir}/fold{i}.csv', index=True)

if __name__ == '__main__':
    data_path = '/home/perry/nvme3n1/TCGA-Brain/All_Features/CONCH/All_Features/'
    tsv_path = 'lgg_tcga_pan_can_atlas_2018_clinical_data.tsv'
    df = pd.read_csv(tsv_path, sep='\t')
    patient_subtype_dict = df.set_index('Patient ID')['Subtype'].to_dict()
    all_names = os.listdir(data_path)
    kv1 = defaultdict(list)
    len1 = len('TCGA-HT-7860')
    
    for _ in all_names:
        kv1[_[:len1]].append(_[:-3])
    for k, v in kv1.items():
        v.sort()
        kv1[k] = v[0]
    common_keys = set(patient_subtype_dict.keys()) & set(kv1.keys())
    task1 = {}
    task2 = {}

    for k in common_keys:
        # task1 基因突变预测
        if patient_subtype_dict[k] == 'LGG_IDHwt':
            task1[kv1[k]] = 0
        elif patient_subtype_dict[k] == 'LGG_IDHmut-non-codel' or patient_subtype_dict[k] == 'LGG_IDHmut-codel':
            task1[kv1[k]] = 1
        # task2 基因缺失预测
        if patient_subtype_dict[k] == 'LGG_IDHmut-codel':
            task2[kv1[k]] = 1
        elif patient_subtype_dict[k] == 'LGG_IDHwt' or patient_subtype_dict[k] == 'LGG_IDHmut-non-codel':
            task2[kv1[k]] = 0
    create_fold(task1, 5, 0.2, 'csv/task1')
    create_fold(task2, 5, 0.2, 'csv/task2')
    c = 1
    pass
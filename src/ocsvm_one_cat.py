import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import copy
import itertools

from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.svm import SVC, OneClassSVM
# How big is train dataset ?
train_test_ratio = 0.039
np.random.RandomState(seed=42)

cats = ["srcip","dstip","proto","state","dur","sbytes","dbytes","sttl","dttl","sloss","dloss","service","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len","Sjit","Djit","Sintpkt","Dintpkt","tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","attack_cat","Label"]

data_path = os.path.abspath(os.path.join(__file__, '..', '..', 'dataset'))

df = pd.read_csv(os.path.join(data_path, "UNSW-NB15_1_anonymized_no_col_names.csv"), names=cats)

# names = pd.read_csv(os.path.join(data_path, 'NUSW-NB15_features_v2.csv'))['Name'].tolist()

# frames = []

# frames.append(pd.read_csv(os.path.join(data_path, "UNSW-NB15_1.csv"), names=names))
# # Uncomment to load all csv
# frames.append(pd.read_csv(os.path.join(data_path, "UNSW-NB15_2.csv"), names=names))
# frames.append(pd.read_csv(os.path.join(data_path, "UNSW-NB15_3.csv"), names=names))
# frames.append(pd.read_csv(os.path.join(data_path, "UNSW-NB15_4.csv"), names=names))

# df = pd.concat(frames, axis=0, ignore_index=True)

mask = np.random.rand(len(df)) < train_test_ratio
train = df[mask]
test = df[~mask]

# Clear memory
del df

train_cats = copy.deepcopy(cats)
# For traing cats Label and attack_cat are removed
train_cats.pop()
train_cats.pop()

del train_cats[train_cats.index('service')] # delete service
del train_cats[train_cats.index('srcip')]  # delete srcip
del train_cats[train_cats.index('dstip')]  # delete dstip
del train_cats[train_cats.index('proto')]  # delete proto
del train_cats[train_cats.index('state')]  # delete state

# del train['service']
# del train['srcip']
# del train['dstip']
# del train['proto']
# del train['state']
# del train['attack_cat']
# del train['dsport']

label_train = copy.deepcopy(train['Label'])
# del train['Label']


tp_arr = []
tn_arr = []
fp_arr = []
fn_arr = []

# del names[names.index('srcip')]
# del names[names.index('dstip')]
# del names[names.index('proto')]
# del names[names.index('state')]
# del names[names.index('service')]
# del names[names.index('Label')]
# del names[names.index('attack_cat')]

pairs = list(itertools.combinations(train_cats, 2))

for cat_pair in pairs:
    print(cat_pair)

    one_class_svm = OneClassSVM(gamma='auto')
    one_class_svm.fit(train[[cat_pair[0], cat_pair[1]]])

    output = one_class_svm.predict(train[[cat_pair[0], cat_pair[1]]])

    new_output = copy.deepcopy(output)
    for count, out in enumerate(output):
        if out == -1:
            new_output[count] = 1
        else:
            new_output[count] = 0

    y_train_output = label_train

    tn, fp, fn, tp = confusion_matrix(y_train_output, new_output).ravel()
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn} <- attacks not detected")

    tp_arr.append(tp)
    tn_arr.append(tn)
    fp_arr.append(fp)
    fn_arr.append(fn)

    with open(f"{os.path.abspath(os.path.join(__file__, '..', 'results', 'ocsvm_anonymize_data.txt'))}", "a") as results_file:
        results_file.write(f"\n{[cat_pair[0], cat_pair[1]]}")
        results_file.write(f"\nTP: {tp}")
        results_file.write(f"\nTN: {tn}")
        results_file.write(f"\nFP: {fp}")
        results_file.write(f"\nFN: {fn} <- attacks not detected")
    

# find index of highest value in array tp_arr and tn_arr and lowest value in array fp_arr and fn_arr
max_tp_index = tp_arr.index(max(tp_arr))
max_tn_index = tn_arr.index(max(tn_arr))
min_fp_index = fp_arr.index(min(fp_arr))
min_fn_index = fn_arr.index(min(fn_arr))

print(f"Categories with highest TP: {pairs[max_tp_index]},\nTP: {tp_arr[max_tp_index]},\nTN: {tn_arr[max_tp_index]},\nFP: {fp_arr[max_tp_index]},\nFN: {fn_arr[max_tp_index]}")
print(f"Categories with highest TN: {pairs[max_tn_index]},\nTP: {tp_arr[max_tn_index]},\nTN: {tn_arr[max_tn_index]},\nFP: {fp_arr[max_tp_index]},\nFN: {fn_arr[max_tn_index]}")
print(f"Categories with lowest FP: {pairs[min_fp_index]},\nTP: {tp_arr[min_fp_index]},\nTN: {tn_arr[min_fp_index]},\nFP: {fp_arr[min_fp_index]},\nFN: {fn_arr[min_fp_index]}")
print(f"Categories with lowest FN: {pairs[min_fn_index]},\nTP: {tp_arr[min_fn_index]},\nTN: {tn_arr[min_fn_index]},\nFP: {fp_arr[min_fn_index]},\nFN: {fn_arr[min_fn_index]}")


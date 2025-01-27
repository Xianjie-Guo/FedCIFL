[Federated Causally Invariant Feature Learning](https://xianjie-guo.github.io/EnHome.html) <br>

# Usage
"FedCIFL.py" is the main function. <br>
----------------------------------------------
```Python
def FedCIFL(Xs_orig: list[np.ndarray], Ys: list[np.ndarray], delta: float = 0.01):
```
* INPUT: <br>
```Python
Xs_orig: a list of datasets on multiple clients (Only supports binary data).
Ys: a list of labels on multiple clients.
delta: the threshold for dividing causal invariant features and irrelevant features.
```
* OUTPUT: <br>
```Python
causal_fea_final: the learned causal invariant features.
```

# Example of experiments on synthetic datasets
```Python
from FedCIFL import *
import time
from tqdm import tqdm
from federated_classifier import federated_classifier_train_test
import numpy as np


def write_metric2txt(Acc: list, Rmse: list, F1: list, path: str):
    acc_mean = np.mean(Acc)
    acc_std = np.std(Acc)
    rmse_mean = np.mean(Rmse)
    rmse_std = np.std(Rmse)
    f1_mean = np.mean(F1)
    f1_std = np.std(F1)

    acc_mean = round(acc_mean, 4)
    acc_std = round(acc_std, 4)
    rmse_mean = round(rmse_mean, 4)
    rmse_std = round(rmse_std, 4)
    f1_mean = round(f1_mean, 4)
    f1_std = round(f1_std, 4)

    with open(path, 'w') as file:
        file.write(f"{acc_mean} {acc_std}\n")
        file.write(f"{rmse_mean} {rmse_std}\n")
        file.write(f"{f1_mean} {f1_std}\n")


num_client_vec = [2, 3, 5, 8, 12, 20]
num_fea_vec = [20, 40]

'''IID+OOD'''
for nfv in num_fea_vec:
    print(f"******************d={nfv}******************")
    for ncv in tqdm(num_client_vec):
        print(f"******************m={ncv}******************")
        Xs = []
        Ys = []
        for i in range(ncv):
            X_train_path = f'../Dataset/IID_OOD_train/train_X_n{nfv}_C{ncv}c{i + 1}.txt'
            Y_train_path = f'../Dataset/IID_OOD_train/train_Y_n{nfv}_C{ncv}c{i + 1}.txt'

            Xs.append(np.loadtxt(X_train_path))
            Y_in = np.loadtxt(Y_train_path)
            Y_in = Y_in.reshape(-1, 1)
            Ys.append(Y_in)

        t_start = time.time()
        ops.reset_default_graph()
        selected_fea = FedCIFL(Xs_orig=Xs, Ys=Ys, delta=0.001)
        total_time = time.time() - t_start

        print(f'selected_features: {selected_fea}')
        print(f'**********FedCIFL running {round(total_time, 10)} s**********')

        Acc_mlp = []
        Rmse_mlp = []
        F1_mlp = []
        Acc_lr = []
        Rmse_lr = []
        F1_lr = []
        for j in range(5):
            X_test_path = f'../Dataset/test/test_X_n{nfv}_v{j + 1}.txt'
            Y_test_path = f'../Dataset/test/test_Y_n{nfv}_v{j + 1}.txt'

            X_test = np.loadtxt(X_test_path)
            Y_test = np.loadtxt(Y_test_path).reshape(-1, 1)

            [accuracy_mlp, rmse_mlp, f1_mlp] = federated_classifier_train_test(Xs_train_orig=Xs, Ys_train=Ys,
                                                                               X_test_orig=X_test,
                                                                               Y_test=Y_test,
                                                                               causal_fea=selected_fea, model='MLP',
                                                                               fl_round_num=300)

            [accuracy_lr, rmse_lr, f1_lr] = federated_classifier_train_test(Xs_train_orig=Xs, Ys_train=Ys,
                                                                            X_test_orig=X_test,
                                                                            Y_test=Y_test,
                                                                            causal_fea=selected_fea, model='LR',
                                                                            fl_round_num=1000)
            Acc_mlp.append(accuracy_mlp)
            Rmse_mlp.append(rmse_mlp)
            F1_mlp.append(f1_mlp)
            Acc_lr.append(accuracy_lr)
            Rmse_lr.append(rmse_lr)
            F1_lr.append(f1_lr)

        out_path1 = f'./results/MLP/FedCIFL_IID_OOD_n{nfv}_C{ncv}.txt'
        out_path2 = f'./results/LR/FedCIFL_IID_OOD_n{nfv}_C{ncv}.txt'
        write_metric2txt(Acc=Acc_mlp, Rmse=Rmse_mlp, F1=F1_mlp, path=out_path1)
        write_metric2txt(Acc=Acc_lr, Rmse=Rmse_lr, F1=F1_lr, path=out_path2)

        print(
            f'Acc_mlp_avg: {round(np.mean(Acc_mlp), 4)}, RMSE_mlp_avg: {round(np.mean(Rmse_mlp), 4)}, F1_mlp_avg: {round(np.mean(F1_mlp), 4)}')
        print(
            f'Acc_lr_avg: {round(np.mean(Acc_lr), 4)}, RMSE_lr_avg: {round(np.mean(Rmse_lr), 4)}, F1_lr_avg: {round(np.mean(F1_lr), 4)}')

        pass
    print("\n\n")
print("\n\n\n")


'''Non-IID+OOD'''
print("******************Non-IID+OOD******************")
for nfv in num_fea_vec:
    print(f"******************d={nfv}******************")
    for ncv in tqdm(num_client_vec):
        print(f"******************m={ncv}******************")
        Xs = []
        Ys = []
        for i in range(ncv):
            X_train_path = f'../Dataset/Non-IID_OOD_train/train_X_n{nfv}_C{ncv}c{i + 1}.txt'
            Y_train_path = f'../Dataset/Non-IID_OOD_train/train_Y_n{nfv}_C{ncv}c{i + 1}.txt'

            Xs.append(np.loadtxt(X_train_path))
            Y_in = np.loadtxt(Y_train_path)
            Y_in = Y_in.reshape(-1, 1)
            Ys.append(Y_in)

        t_start = time.time()
        ops.reset_default_graph()
        selected_fea = FedCIFL(Xs_orig=Xs, Ys=Ys, delta=0.001)
        total_time = time.time() - t_start

        print(f'selected_features: {selected_fea}')
        print(f'**********FedCIFL running {round(total_time, 10)} s**********')

        Acc_mlp = []
        Rmse_mlp = []
        F1_mlp = []
        Acc_lr = []
        Rmse_lr = []
        F1_lr = []
        for j in range(5):
            X_test_path = f'../Dataset/test/test_X_n{nfv}_v{j + 1}.txt'
            Y_test_path = f'../Dataset/test/test_Y_n{nfv}_v{j + 1}.txt'

            X_test = np.loadtxt(X_test_path)
            Y_test = np.loadtxt(Y_test_path).reshape(-1, 1)

            [accuracy_mlp, rmse_mlp, f1_mlp] = federated_classifier_train_test(Xs_train_orig=Xs, Ys_train=Ys,
                                                                               X_test_orig=X_test,
                                                                               Y_test=Y_test,
                                                                               causal_fea=selected_fea, model='MLP',
                                                                               fl_round_num=300)

            [accuracy_lr, rmse_lr, f1_lr] = federated_classifier_train_test(Xs_train_orig=Xs, Ys_train=Ys,
                                                                            X_test_orig=X_test,
                                                                            Y_test=Y_test,
                                                                            causal_fea=selected_fea, model='LR',
                                                                            fl_round_num=1000)
            Acc_mlp.append(accuracy_mlp)
            Rmse_mlp.append(rmse_mlp)
            F1_mlp.append(f1_mlp)
            Acc_lr.append(accuracy_lr)
            Rmse_lr.append(rmse_lr)
            F1_lr.append(f1_lr)

        out_path1 = f'./results/MLP/FedCIFL_Non-IID_OOD_n{nfv}_C{ncv}.txt'
        out_path2 = f'./results/LR/FedCIFL_Non-IID_OOD_n{nfv}_C{ncv}.txt'
        write_metric2txt(Acc=Acc_mlp, Rmse=Rmse_mlp, F1=F1_mlp, path=out_path1)
        write_metric2txt(Acc=Acc_lr, Rmse=Rmse_lr, F1=F1_lr, path=out_path2)

        print(
            f'Acc_mlp_avg: {round(np.mean(Acc_mlp), 4)}, RMSE_mlp_avg: {round(np.mean(Rmse_mlp), 4)}, F1_mlp_avg: {round(np.mean(F1_mlp), 4)}')
        print(
            f'Acc_lr_avg: {round(np.mean(Acc_lr), 4)}, RMSE_lr_avg: {round(np.mean(Rmse_lr), 4)}, F1_lr_avg: {round(np.mean(F1_lr), 4)}')

        pass
    print("\n\n")
print("\n\n\n")
```

# Import package
```Python
Package                      Version
---------------------------- ------------
absl-py                      2.1.0
astunparse                   1.6.3
cachetools                   5.3.3
certifi                      2024.2.2
charset-normalizer           3.3.2
flatbuffers                  24.3.25
gast                         0.4.0
google-auth                  2.29.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.63.0
h5py                         3.11.0
idna                         3.7
importlib_metadata           7.1.0
joblib                       1.4.2
keras                        2.11.0
libclang                     18.1.1
Markdown                     3.6
MarkupSafe                   2.1.5
numpy                        1.26.4
oauthlib                     3.2.2
opt-einsum                   3.3.0
packaging                    24.0
pillow                       10.3.0
pip                          23.3.1
protobuf                     3.19.6
pyasn1                       0.6.0
pyasn1_modules               0.4.0
requests                     2.31.0
requests-oauthlib            2.0.0
rsa                          4.9
scikit-learn                 1.4.2
scipy                        1.13.0
setuptools                   68.2.2
six                          1.16.0
tensorboard                  2.11.2
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow-estimator         2.11.0
tensorflow-gpu               2.11.0
tensorflow-io-gcs-filesystem 0.37.0
termcolor                    2.4.0
threadpoolctl                3.5.0
torch                        1.12.1+cu113
torchaudio                   0.12.1+cu113
torchvision                  0.13.1+cu113
tqdm                         4.66.4
typing_extensions            4.11.0
urllib3                      2.2.1
Werkzeug                     3.0.2
wheel                        0.43.0
wrapt                        1.16.0
zipp                         3.18.1
```

# Reference
* Guo, Xianjie, et al. "Federated Causally Invariant Feature Learning." *Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI'25)* (2025).

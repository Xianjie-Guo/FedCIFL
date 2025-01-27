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
            X_train_path = f'../data-generation/STV/IID_OOD_train/train_X_n{nfv}_C{ncv}c{i + 1}.txt'
            Y_train_path = f'../data-generation/STV/IID_OOD_train/train_Y_n{nfv}_C{ncv}c{i + 1}.txt'

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
            X_test_path = f'../data-generation/STV/test/test_X_n{nfv}_v{j + 1}.txt'
            Y_test_path = f'../data-generation/STV/test/test_Y_n{nfv}_v{j + 1}.txt'

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
            X_train_path = f'../data-generation/STV/Non-IID_OOD_train/train_X_n{nfv}_C{ncv}c{i + 1}.txt'
            Y_train_path = f'../data-generation/STV/Non-IID_OOD_train/train_Y_n{nfv}_C{ncv}c{i + 1}.txt'

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
            X_test_path = f'../data-generation/STV/test/test_X_n{nfv}_v{j + 1}.txt'
            Y_test_path = f'../data-generation/STV/test/test_Y_n{nfv}_v{j + 1}.txt'

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

import random
import torch
import numpy as np
import copy


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=output_dim, bias=True),
            torch.nn.Sigmoid(),
        )
        self.criterion = torch.nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        self.optimizer = torch.optim.Adam(self.parameters())

        self.training_loss = []
        self.validation_loss = []
        self.accuracies = []
        self.f1_scores = []

    def forward(self, x):
        return self.layers(x)

    def Training(self, input: torch.Tensor, label: torch.Tensor):
        output = self.forward(input)
        loss = self.criterion(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_loss.append(loss.item())

    def evaluation(self, input: torch.Tensor, label: torch.Tensor):
        with torch.no_grad():
            output = self.forward(input)

            tp = ((torch.max(output.data, dim=1)[1] == torch.max(label.data, dim=1)[1]) * (
                    torch.max(output.data, dim=1)[1] == 1)).sum().item()
            fp = ((torch.max(output.data, dim=1)[1] != torch.max(label.data, dim=1)[1]) * (
                    torch.max(output.data, dim=1)[1] == 1)).sum().item()
            fn = ((torch.max(output.data, dim=1)[1] != torch.max(label.data, dim=1)[1]) * (
                    torch.max(output.data, dim=1)[1] == 0)).sum().item()

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            rmse = torch.pow(self.criterion(output, label), 0.5).item()
            acc = ((torch.max(output.data, dim=1)[1] == torch.max(label.data, dim=1)[1]).sum().item() / label.shape[0])

            if precision + recall == 0:
                f_measure = 0
            else:
                f_measure = (2 * precision * recall) / (precision + recall)

            self.validation_loss.append(rmse)
            self.accuracies.append(acc)
            self.f1_scores.append(f_measure)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=150, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=150, out_features=output_dim, bias=True),
            torch.nn.Sigmoid(),
        )
        self.criterion = torch.nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        self.optimizer = torch.optim.Adam(self.parameters())

        self.training_loss = []
        self.validation_loss = []
        self.accuracies = []
        self.f1_scores = []

    def forward(self, x):
        return self.layers(x)

    def Training(self, input: torch.Tensor, label: torch.Tensor):
        output = self.forward(input)
        loss = self.criterion(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_loss.append(loss.item())

    def evaluation(self, input: torch.Tensor, label: torch.Tensor):
        with torch.no_grad():
            output = self.forward(input)

            tp = ((torch.max(output.data, dim=1)[1] == torch.max(label.data, dim=1)[1]) * (
                        torch.max(output.data, dim=1)[1] == 1)).sum().item()
            fp = ((torch.max(output.data, dim=1)[1] != torch.max(label.data, dim=1)[1]) * (
                        torch.max(output.data, dim=1)[1] == 1)).sum().item()
            fn = ((torch.max(output.data, dim=1)[1] != torch.max(label.data, dim=1)[1]) * (
                        torch.max(output.data, dim=1)[1] == 0)).sum().item()

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            rmse = torch.pow(self.criterion(output, label), 0.5).item()
            acc = ((torch.max(output.data, dim=1)[1] == torch.max(label.data, dim=1)[1]).sum().item() / label.shape[0])

            if precision + recall == 0:
                f_measure = 0
            else:
                f_measure = (2 * precision * recall) / (precision + recall)

            self.validation_loss.append(rmse)
            self.accuracies.append(acc)
            self.f1_scores.append(f_measure)


def federated_classifier_train_test(Xs_train_orig: list[np.ndarray], Ys_train: list[np.ndarray], X_test_orig: np.ndarray,
                                    Y_test: np.ndarray, causal_fea: list[int] = None, model: str = 'MLP',
                                    class_num: int = 2, fl_round_num: int = 300, local_iteration_num: int = 20,
                                    local_batch_size: int = 100):

    seed = 888
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    client_num = len(Xs_train_orig)

    Xs_train = []

    for index_client in range(client_num):
        Xs_train.append(np.copy(Xs_train_orig[index_client]))
    X_test = np.copy(X_test_orig)

    # 根据因果特征子集降维原始特征集
    for index_client in range(client_num):
        Xs_train[index_client] = Xs_train[index_client][:, causal_fea]
    X_test = X_test[:, causal_fea]

    input_dim = Xs_train[0].shape[1]
    output_dim = class_num

    if model == 'LR':
        global_model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    elif model == 'MLP':
        global_model = MultiLayerPerceptron(input_dim=input_dim, output_dim=output_dim)
    local_models = [copy.deepcopy(global_model) for i in range(0, client_num)]
    for i in range(0, client_num):
        local_models[i].train()

    xs_train = [torch.tensor(Xs_train[i], dtype=torch.float, device='cuda') for i in range(0, client_num)]
    x_test = torch.tensor(X_test, dtype=torch.float, device='cuda')

    ys_train = [torch.nn.functional.one_hot(torch.tensor(Ys_train[i].squeeze(), dtype=torch.long, device='cuda'),
                                            num_classes=class_num).float() for i in range(0, client_num)]
    y_test = torch.nn.functional.one_hot(torch.tensor(Y_test.squeeze(), dtype=torch.long, device='cuda'),
                                         num_classes=class_num).float()

    local_datasets_size = [int(Xs_train[i].shape[0]) for i in range(0, client_num)]
    weights = [local_datasets_size[i] / sum(local_datasets_size) for i in range(0, client_num)]

    model_keys = local_models[0].state_dict().keys()

    for _ in range(0, fl_round_num):
        for i in range(0, client_num):
            local_models[i].load_state_dict(copy.deepcopy(global_model.state_dict()))

        for i in range(0, client_num):
            local_models[i].to('cuda')
            for __ in range(0, local_iteration_num):
                # print(local_datasets_size)
                random_index = torch.tensor(random.sample(range(0, local_datasets_size[i]), local_batch_size),
                                            dtype=torch.long, device='cuda')
                input = torch.index_select(xs_train[i], index=random_index, dim=0)
                label = torch.index_select(ys_train[i], index=random_index, dim=0)
                local_models[i].Training(input, label)
            local_models[i].to('cpu')

        global_state_dict = {}
        for key in model_keys:
            temp = 0
            for i in range(0, client_num):
                temp += local_models[i].state_dict()[key].clone().detach() * weights[i]
            global_state_dict[key] = temp.clone().detach()

        global_model.load_state_dict(copy.deepcopy(global_state_dict))
        global_model.to('cuda')
        global_model.evaluation(x_test, y_test)
        global_model.to('cpu')
        # print(f'Round {_} done!')
        # print(f'Train Loss: {local_models[0].training_loss[-1]}')
        # print(f'ACC: {global_model.accuracies[-1]}')
        # print(f'RMSE: {global_model.validation_loss[-1]}')
        # print(f'F-1: {global_model.f1_scores[-1]}')
        # print('')

    return global_model.accuracies[-1], global_model.validation_loss[-1], global_model.f1_scores[-1]

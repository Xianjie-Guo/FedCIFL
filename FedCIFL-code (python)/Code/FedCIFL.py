from com_funcs import *
from tensorflow.python.framework import ops

# tf.set_random_seed(888)  # reproducibility


def FedCIFL(Xs_orig: list[np.ndarray], Ys: list[np.ndarray], delta: float = 0.01):

    num_client = len(Xs_orig)

    Xs = []

    for index_client in range(num_client):
        Xs.append(np.copy(Xs_orig[index_client]))

    iter = 0

    while(1):
        # Step 1: Learning Sample Weights and Estimating Causal Effects
        # print('Step 1: Learning Sample Weights and Estimating Causal Effects ...')

        betas = []
        min_losses = []
        for index_client in range(num_client):
            learning_rate = 0.01
            num_steps = 2000
            tol = 1e-8
            ops.reset_default_graph()
            # learn autoencoder by deep logistic regression with non-reweighting samples
            _, min_loss = f_auto_encoder(1, 0, Xs[index_client], Ys[index_client], learning_rate, num_steps, tol)
            min_losses.append(min_loss)

            n, d = Xs[index_client].shape
            X_all = Xs[index_client]
            for j in range(d):
                X_j = np.copy(Xs[index_client])
                X_j[:, j] = 0
                X_all = np.vstack((X_all, X_j))

            ops.reset_default_graph()
            X_all_encoder, _ = f_auto_encoder(0, 0, X_all, Ys[index_client], 0, 0, 0)
            X_all_encoder[0] = map_to_interval(A=X_all_encoder[0], k=6)  # 将每列等分成k段

            # global sample weights learning by global balancing on embedded confounders
            ops.reset_default_graph()
            learning_rate = 0.005
            num_steps = 4000
            tol = 1e-8
            GG = f_global_balancing(1, Xs[index_client], X_all_encoder[0], learning_rate, num_steps, tol)

            # retaining preditive model by deep logistic regression with reweighted samples
            learning_rate = 0.005
            num_steps = 4000
            tol = 1e-8
            ops.reset_default_graph()
            RMSE, F1, beta = f_logistic_regression_weighted(1, Xs[index_client], GG[0], Ys[index_client], learning_rate, num_steps, tol)

            betas.append(abs(beta[0]))

        # Step 2: Sending Potential Irrelevant Features and Causal Effects
        siirs = []
        for index_client in range(num_client):
            causal_fea_index = fea_filter_by_threshold(betas[index_client], delta)
            all_fea_index = set(range(len(betas[index_client])))
            selected_fea_index = set(causal_fea_index)
            remaining_fea_index = list(all_fea_index - selected_fea_index)
            siirs.append(remaining_fea_index)

        # Step 3: Determining the Optimal Irrelevant Feature Set
        num_siirs = []
        for index_client in range(num_client):
            num_siirs.append(len(siirs[index_client]))

        sorted_indices = np.argsort(min_losses)
        num_siir_optimal = vote_constant(num_siirs, sorted_indices)  # weighted voting

        if num_siir_optimal==0:
            break

        iter = iter + 1

        betas_array = np.array(betas)
        beta_sum = np.sum(betas_array, axis=0)

        siir_optimal = np.argsort(beta_sum, axis=0)[:num_siir_optimal]

        # Step 4: Sending the Latest Confounders and Updating Local Data
        for index_client in range(num_client):
            mask = np.ones(Xs[index_client].shape[1], dtype=bool)
            mask[siir_optimal] = False
            Xs[index_client] = Xs[index_client][:, mask]

    causal_fea_final = []
    for i in range(Xs[0].shape[1]):
        for j in range(Xs_orig[0].shape[1]):
            if np.array_equal(Xs[0][:, i], Xs_orig[0][:, j]):
                causal_fea_final.append(j)
                break

    return causal_fea_final

import numpy as np
import sklearn.metrics as skm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR')
tf.disable_v2_behavior()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


#######################################################################################################################
def f_logistic_regression(train_or_test, X_in, Y_in, learning_rate, num_steps, tol):
    tf.set_random_seed(888)  # reproducibility
    n, num_feature = X_in.shape

    display_step = 500

    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, num_feature])
    Y = tf.placeholder("float", [None, 1])

    # prediction
    W = tf.Variable(tf.random_normal([num_feature, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.nn.sigmoid(tf.matmul(X, W) + b)

    saver = tf.train.Saver()
    sess = tf.Session()

    if train_or_test == 1:
        # print('Logistic_regression...training...start...')
        loss_predictive = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(hypothesis, 1e-8, 1.0)) + (1 - Y) * tf.log(
            tf.clip_by_value(1 - hypothesis, 1e-8, 1.0)))
        loss_l2reg = tf.reduce_sum(tf.abs(W))
        loss = loss_predictive + 0.01 * loss_l2reg

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l2reg = sess.run([optimizer, loss, loss_l2reg], feed_dict={X: X_in, Y: Y_in})
            if abs(l - l_pre) <= tol:
                # print('Converge ... Step %i: Minibatch Loss: %f ... %f' % (i, l, l2reg))
                break
            l_pre = l
            # if i % display_step == 0 or i == 1:
            #     print('Step %i: Minibatch Loss: %f ... %f' % (i, l, l2reg))
        if not os.path.isdir('models/logistic_regression/'):
            os.makedirs('models/logistic_regression/')
        saver.save(sess, 'models/logistic_regression/logistic_regression.ckpt')
    else:
        # print('Logistic_regression...testing...start...')
        saver.restore(sess, 'models/logistic_regression/logistic_regression.ckpt')

    RMSE = tf.sqrt(tf.reduce_mean((Y - hypothesis) ** 2))
    RMSE_error, Y_predict = sess.run([RMSE, hypothesis], feed_dict={X: X_in, Y: Y_in})
    F1_score = skm.f1_score(Y_in, Y_predict > 0.5)

    sess.close()
    return RMSE_error, F1_score


#######################################################################################################################
def f_auto_encoder(train_or_test, predict_or_not, X_input, Y_input, learning_rate, num_steps, tol):
    tf.set_random_seed(888)  # reproducibility
    n, num_feature = X_input.shape

    num_hidden_1 = int(num_feature / 2)  # 1st layer num features
    num_hidden_2 = int(num_feature / 2)  # 2nd layer num features (the latent dim)

    display_step = 500

    X = tf.placeholder("float", [None, num_feature])
    Y = tf.placeholder("float", [None, 1])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_feature, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_feature])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([num_feature])),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # encoder of X
    X_encoder = encoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # prediction
    W = tf.Variable(tf.random_normal([num_hidden_2, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.sigmoid(tf.matmul(X_encoder, W) + b)

    saver = tf.train.Saver()
    sess = tf.Session()

    min_loss = float('inf')

    if train_or_test == 1:
        # print('Auto_encoder...training...start...')
        loss_autoencoder = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        # loss_predictive = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
        loss_predictive = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(hypothesis, 1e-8, 1.0)) + (1 - Y) * tf.log(
            tf.clip_by_value(1 - hypothesis, 1e-8, 1.0)))
        # loss_predictive = -tf.reduce_sum(tf.divide((G*G)*(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)),tf.reduce_sum(G*G)))
        loss_l2reg = tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(weights['encoder_h1'])) + tf.reduce_sum(
            tf.square(weights['encoder_h2'])) + tf.reduce_sum(tf.square(weights['decoder_h1'])) + tf.reduce_sum(
            tf.square(weights['decoder_h2']))

        # loss_predictive是交叉熵损失
        # loss_autoencoder是重构损失
        # loss_l2reg是正则化项
        loss = 10 * loss_predictive + 1 * loss_autoencoder + 0.0001 * loss_l2reg
        # loss = 10 * loss_predictive + 10.0 / num_hidden_2 * loss_autoencoder + 0.0001 * loss_l2reg
        # loss = 1.0 * loss_predictive + 10.0 / num_hidden_2 * loss_autoencoder + 0.00001 * loss_l2reg
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l_predictive, l_autoencoder, l_l2reg = sess.run(
                [optimizer, loss, loss_predictive, loss_autoencoder, loss_l2reg], feed_dict={X: X_input, Y: Y_input})

            min_loss = min(min_loss, l)

            if abs(l - l_pre) <= tol:
                # print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (
                #     i, l, l_predictive, l_autoencoder, l_l2reg))
                break
            l_pre = l
            # if i % display_step == 0 or i == 1:
                # print('Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (i, l, l_predictive, l_autoencoder, l_l2reg))

        if not os.path.isdir('models/auto_encoder/'):
            os.makedirs('models/auto_encoder/')
        saver.save(sess, 'models/auto_encoder/auto_encoder.ckpt')
    else:
        # print('Auto_encoder...testing...start...')
        saver.restore(sess, 'models/auto_encoder/auto_encoder.ckpt')

    if predict_or_not == 1:
        RMSE = tf.sqrt(tf.reduce_mean((Y - hypothesis) ** 2))
        RMSE_error, Y_predict = sess.run([RMSE, hypothesis], feed_dict={X: X_input, Y: Y_input})
        F1_score = skm.f1_score(Y_input, Y_predict > 0.5)
        return RMSE_error, F1_score

    return sess.run([X_encoder], feed_dict={X: X_input}), min_loss


#######################################################################################################################
def f_global_balancing(train_or_test, X_input, X_encoder_input, learning_rate, num_steps, tol):
    tf.set_random_seed(888)  # reproducibility
    n, d = X_input.shape
    n_e, d_e = X_encoder_input.shape

    display_step = 500

    X = tf.placeholder("float", [None, d])
    X_encoder = tf.placeholder("float", [None, d_e])

    G = tf.Variable(tf.ones([n, 1]))

    loss_balancing = tf.constant(0, tf.float32)
    for j in range(1, d + 1):
        X_j = tf.slice(X_encoder, [j * n, 0], [n, d_e])
        I = tf.slice(X, [0, j - 1], [n, 1])
        balancing_j = tf.divide(tf.matmul(tf.transpose(X_j), G * G * I),
                                tf.maximum(tf.reduce_sum(G * G * I), tf.constant(0.1))) - tf.divide(
            tf.matmul(tf.transpose(X_j), G * G * (1 - I)), tf.maximum(tf.reduce_sum(G * G * (1 - I)), tf.constant(0.1)))

        loss_balancing += tf.norm(balancing_j, ord=2)
    loss_regulizer = (tf.reduce_sum(G * G) - n) ** 2 + 10 * (tf.reduce_sum(G * G - 1)) ** 2  #

    loss = loss_balancing + 0.0001 * loss_regulizer

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    X_feed = X_input
    for j in range(d):
        X_j = np.copy(X_input)
        X_j[:, j] = 0
        X_feed = np.vstack((X_feed, X_j))

    if train_or_test == 1:
        # print('Global_balancing...training...start...')
        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l_balancing, l_regulizer = sess.run([optimizer, loss, loss_balancing, loss_regulizer],
                                                      feed_dict={X: X_input, X_encoder: X_encoder_input})
            if abs(l - l_pre) <= tol:
                # print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_balancing, l_regulizer))
                break
            l_pre = l
            if l_balancing < 0.05:
                # print('Good enough ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_balancing, l_regulizer))
                break
            # if i % display_step == 0 or i == 1:
                # print('Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_balancing, l_regulizer))
        if not os.path.isdir('models/global_balancing/'):
            os.makedirs('models/global_balancing/')
        saver.save(sess, 'models/global_balancing/global_balancing.ckpt')
    else:
        # print('Global_balancing...testing...start...')
        saver.restore(sess, 'models/global_balancing/global_balancing.ckpt')

    return sess.run([G], feed_dict={X: X_input, X_encoder: X_encoder_input})


#######################################################################################################################
def f_logistic_regression_weighted(train_or_test, X_in, G_input, Y_in, learning_rate, num_steps, tol):
    tf.set_random_seed(888)  # reproducibility
    n, num_feature = X_in.shape
    display_step = 500
    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, num_feature])
    Y = tf.placeholder("float", [None, 1])
    G = tf.placeholder("float", [None, 1])

    # prediction
    W = tf.Variable(tf.random_normal([num_feature, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.nn.sigmoid(tf.matmul(X, W) + b)

    saver = tf.train.Saver()
    sess = tf.Session()

    if train_or_test == 1:
        # print('Logistic_regression_weighted...training...start...')
        loss_predictive = -tf.reduce_sum(tf.divide(G * G * (
                Y * tf.log(tf.clip_by_value(hypothesis, 1e-8, 1)) + (1 - Y) * tf.log(
            tf.clip_by_value(1 - hypothesis, 1e-8, 1))), tf.reduce_sum(G * G)))

        loss_l2reg = tf.reduce_sum(tf.abs(W))
        # loss = loss_predictive + 0.01 * loss_l2reg
        loss = loss_predictive + 0.01 * loss_l2reg

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l2reg = sess.run([optimizer, loss, loss_l2reg], feed_dict={X: X_in, Y: Y_in, G: G_input})
            if abs(l - l_pre) <= tol:
                # print('Converge ... Step %i: Minibatch Loss: %f ... %f' % (i, l, l2reg))
                break
            l_pre = l
            # if i % display_step == 0 or i == 1:
                # print('Step %i: Minibatch Loss: %f ... %f' % (i, l, l2reg))
        if not os.path.isdir('models/logistic_regression_weighted/'):
            os.makedirs('models/logistic_regression_weighted/')
        saver.save(sess, 'models/logistic_regression_weighted/logistic_regression_weighted.ckpt')
    else:
        # print('Logistic_regression_weighted...testing...start...')
        saver.restore(sess, 'models/logistic_regression_weighted/logistic_regression_weighted.ckpt')

    RMSE = tf.sqrt(tf.reduce_mean((Y - hypothesis) ** 2))
    RMSE_error, Y_predict = sess.run([RMSE, hypothesis], feed_dict={X: X_in, Y: Y_in, G: G_input})
    F1_score = skm.f1_score(Y_in, Y_predict > 0.5)
    beta = sess.run([W], feed_dict={X: X_in, Y: Y_in, G: G_input})
    # if train_or_test == 1:
    #     print(beta)
    sess.close()
    return RMSE_error, F1_score, beta


#######################################################################################################################
def fea_filter_by_threshold(beta, delta):
    fea = [x for x in beta]
    causal_fea = []
    for i in range(len(fea)):
        if beta[i] > delta:
            causal_fea.append(i)
    causal_fea = [int(x) for x in causal_fea]
    return np.array(causal_fea)


#######################################################################################################################
def vote_constant(num_siirs, sorted_indices):
    unique_constants, counts = np.unique(num_siirs, return_counts=True)
    max_count = np.max(counts)

    candidates = unique_constants[counts == max_count]
    if len(candidates) == 1:
        return candidates[0]

    avg_ranks = []
    for candidate in candidates:
        indices = np.where(num_siirs == candidate)[0]
        ranks = sorted_indices[indices]
        avg_rank = np.mean(ranks)
        avg_ranks.append(avg_rank)

    min_avg_rank = np.min(avg_ranks)
    winner = candidates[avg_ranks.index(min_avg_rank)]

    return winner


#######################################################################################################################
def map_to_interval(A: np.ndarray, k: int = 6):
    interval_size = 1.0 / k
    interval_bounds = np.arange(0, 1 + interval_size, interval_size)

    A_mapped = np.zeros_like(A)
    for j in range(A.shape[1]):
        for i in range(A.shape[0]):
            val = A[i, j]
            idx = int(val // interval_size)
            if val - interval_bounds[idx] < interval_size / 2:
                A_mapped[i, j] = interval_bounds[idx]
            else:
                A_mapped[i, j] = interval_bounds[idx + 1]

    return A_mapped

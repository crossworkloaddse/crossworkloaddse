import numpy as np
from sklearn import preprocessing

from config import DOMAIN_ENCODE_ID
from get_basic_model import get_basic_model
from get_real_pareto_frontier import response_transfer, response_transfer_inverse
from metric_function import feature_transfer


def model_program_specific(train_data, X_test, surrogate_model_config, metric_name, log_file=None, print_info=False, Y_real=None):

    learner = surrogate_model_config['learner']
    base_estimator = get_basic_model(learner, metric_name=metric_name, return_config=False)

    X_src, Y_src, X_dst, Y_dst = train_data
    X_dst = np.asarray(X_dst)

    response_transfer_method = surrogate_model_config['y_transform']
    if '' != response_transfer_method:
        Y_dst_transform, transfer_param = response_transfer(y=Y_dst, method=response_transfer_method)
    else:
        Y_dst_transform = Y_dst

    if surrogate_model_config['src_use']:
        #min_max_scaler = preprocessing.MinMaxScaler()
        #min_max_scaler2 = preprocessing.MinMaxScaler()

        X_train = None
        Y_train = None
        for src_iter in range(len(Y_src)):
            X_train_iter = feature_transfer(np.asarray(X_src[src_iter]))
            # X_train_iter = X_src[src_iter]
            X_train = np.append(X_train, X_train_iter, axis=0) if X_train is not None else X_train_iter
            Y_train = np.append(Y_train, Y_src[src_iter], axis=0) if Y_train is not None else Y_src[src_iter]
        dst_domain_trans = feature_transfer(np.asarray(X_dst))
        # dst_domain_trans = np.asarray(X_dst)
        X_train = np.append(X_train, dst_domain_trans, axis=0)
        Y_train = np.append(Y_train, Y_dst, axis=0)

        from program_inherent_similarity import get_domain_distance
        #distance_value = get_domain_distance(Y_dst, Y_src[0], domain_loss='wasserstein').item()
        distance_value = 0

        # X_train = min_max_scaler.fit_transform(X_train)
        # Y_train = min_max_scaler2.fit_transform(Y_train.reshape(-1, 1)).ravel()

        X_test = feature_transfer(np.asarray(X_test))
        # X_test_trans = min_max_scaler.transform(X_test_trans)

        base_estimator.fit(X_train, Y_train)
        #predict
        Y_test = base_estimator.predict(X_test)
    else:
        base_estimator.fit(X_dst[:, DOMAIN_ENCODE_ID + 1:], Y_dst_transform)
        #predict
        Y_test = base_estimator.predict(np.asarray(X_test)[:, DOMAIN_ENCODE_ID + 1:])
        distance_value = 0

    if '' != response_transfer_method:
        Y_test = response_transfer_inverse(Y_test, method=response_transfer_method, transfer_param=transfer_param)
    return Y_test, distance_value
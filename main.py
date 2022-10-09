import copy
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import random

from config import *
from get_real_pareto_frontier import transfer_version_to_var
from model_MPGM import model_MPGM
from simulation_metrics import read_metrics, problem_space, read_metrics_cases


def gen_train_data_by_case(case_name, metrics_all, n_sample, test_split=False, random_seed=None):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    domain_id = get_domain_id(case_name)
    #print(f"{case_name} -> {domain_id}")
    if -1 == n_sample:
        # n_sample = len(metrics_all)
        train_instance = metrics_all
    else:
        random.seed(random_seed)
        random.shuffle(metrics_all)
        train_instance = metrics_all[:n_sample]
        if test_split:
            test_instance = metrics_all[n_sample:]
            for train_index, design_point in enumerate(test_instance):
                if len(design_point['version']) < 30:
                    continue
                else:
                    x = transfer_version_to_var(design_point['version'])
                    x_trans = [domain_id]
                    for space_trans, space_var in zip(problem_space, x):
                        x_trans.append(space_trans.transform(space_var))
                    X_test.append(x_trans)
                    Y_test.append(design_point[metric_name])
    for train_index, design_point in enumerate(train_instance):
        if len(design_point['version']) < 30:
            continue
        else:
            x = transfer_version_to_var(design_point['version'])
            x_trans = [domain_id]
            for space_trans, space_var in zip(problem_space, x):
                x_trans.append(space_trans.transform(space_var))
            X_train.append(x_trans)
            Y_train.append(design_point[metric_name])
            # print(f"x={x}")
    #print(f"{case_name}: all size = {len(metrics_all)} instance size= {len(X_train)}")
    if test_split:
        return X_train, Y_train, X_test, Y_test
    else:
        return X_train, Y_train


def gen_train_data(case_name, metrics_all, src_domain_list, surrogate_model_tag, random_seed):
    X_dst, Y_dst, X_test, Y_test = gen_train_data_by_case(case_name=case_name, metrics_all=metrics_all[case_name], n_sample=N_SAMPLES_INIT, test_split=True, random_seed=random_seed)
    src_domain_case_names = src_domain_list
    X_src = [[] for i in range(N_SRC_DOMAIN)]
    Y_src = [[] for i in range(N_SRC_DOMAIN)]
    for src_iter, src_domain_case_name in enumerate(src_domain_case_names):
        if 'PACT07' == surrogate_model_tag or 'ICCD09' == surrogate_model_tag:
            n_sample = N_SAMPLES_ALL
        else:
            n_sample = N_SRC_DOMAIN_TRAIN
        X_src_iter, Y_src_iter = gen_train_data_by_case(case_name=src_domain_case_name, metrics_all=metrics_all[src_domain_case_name], n_sample=n_sample, random_seed=random_seed)
        X_src[src_iter] = X_src_iter
        Y_src[src_iter] = Y_src_iter
    # print(f"gen_train_data X={X}")
    X_src = np.asarray(X_src)
    Y_src = np.asarray(Y_src)
    train_data = X_src, Y_src, X_dst, Y_dst
    return train_data, X_test, Y_test


def evaluate(Y_real, Y_pred):
    # Pearson correlation coefficient
    # Y_pred = Y_pred.reshape(len(Y_pred), 1)
    # Y_real = Y_real.reshape(len(Y_real), 1)
    #print(f"Y_real={Y_real}, Y_pred={Y_pred}")
    #pccs2 = np.corrcoef(Y_pred, Y_real)
    # from scipy.stats import pearsonr
    # pccs = pearsonr(Y_pred, Y_real)
    try:
        from metric_function import cal_pccs
        pccs = cal_pccs(np.asarray(Y_pred), np.asarray(Y_real))
        npccs = 1 - pccs
        if npccs < 0:
            npccs = -1
    except:
        npccs = 1
    #print(f"pccs={pccs}")
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    mae = mean_absolute_error(y_true=Y_real, y_pred=Y_pred)
    mape = mean_absolute_percentage_error(y_true=Y_real, y_pred=Y_pred)
    #print(f"mae={mae}")
    return npccs, mae, mape

def ci(y):
    # 95% for 1.96
    return 1.96 * y.std(axis=0) / np.sqrt(len(y))

exp_id = None
n_experiment = 1
print_info = 0
iter_analysis = False
transfer_analysis = True

if smoke_test:
    surrogate_model_tag_list = ["smoke_test"]
    n_experiment = 1
elif 2 < len(sys.argv):
    # run all models
    surrogate_model_tag_list = [
        #'program_specific',
        #"PACT07",
        #"ICCD09",
        #"EAC",
        #"M5P",
        #"TrEE",
        #"TrDSE",
        #"CASH",
        #"our",
        #"our2",
        #"our4_base_lw",
        "our4_hbo_lw",
        #"our4_hbo_lw_sdv4",
        #"our4_lw_sdv4",
        #"our4_lw_sdv3",
        #"our4",
        #"MPGM",
    ]
    n_experiment = 10
    print_info = 0
    iter_analysis = False
elif "SC-202005121725" == hostname:
    # bookpad
    surrogate_model_tag_list = ["program_specific"]
    n_experiment = 1
    print_info = 1
    iter_analysis = True
elif "DESKTOP-A4P9S3E" == hostname:
    # small destop
    surrogate_model_tag_list = ["our4_lw_sdv4"]
    n_experiment = 10
    print_info = 0
    iter_analysis = False
else:
    surrogate_model_tag_list = ["our4_lw_sdv4"]
    n_experiment = 10
    print_info = 0
    iter_analysis = False


def get_surrogate_model(surrogate_model_tag):
    surrogate_model_tag_real = surrogate_model_tag
    surrogate_model_dict = {}
    surrogate_model_dict['src_use'] = 0  # 1 only for program-specific
    surrogate_model_dict['y_transform'] = ''
    #surrogate_model_dict['y_transform'] = 'box-cox'
    #surrogate_model_dict['y_transform'] = 'StandardScaler'
    match surrogate_model_tag:
        case "smoke_test":
            surrogate_model_dict['v'] = 1
        case "PACT07":
            surrogate_model_dict['v'] = 3
        case "ICCD09":
            surrogate_model_dict['v'] = 3
        case "EAC":
            surrogate_model_dict['v'] = 1
        case "M5P":
            surrogate_model_dict['v'] = 0
        case "TrDSE":
            surrogate_model_dict['v'] = 12
            surrogate_model_dict['S'] = 10
            surrogate_model_tag_real += '_S' + str(surrogate_model_dict['S'])
            surrogate_model_tag_real += '_feat'
        case "TrEE":
            surrogate_model_dict['v'] = 4
        case "MPGM":
            surrogate_model_dict['v'] = 1
        case "CASH":
            surrogate_model_dict['v'] = 0
        case 'program_specific':
            surrogate_model_dict['v'] = 1
            #surrogate_model_dict['learner'] = 'SVR'
            #surrogate_model_dict['learner'] = 'M5P'
            #surrogate_model_dict['learner'] = 'sklearnmlp10'
            #surrogate_model_dict['learner'] = 'sklearnmlp'
            #surrogate_model_dict['learner'] = 'GBRT'
            #surrogate_model_dict['learner'] = 'ET'
            #surrogate_model_dict['learner'] = 'RF'
            #surrogate_model_dict['learner'] = 'AdaGBRT'
            surrogate_model_dict['learner'] = 'AdaMLP'
            #surrogate_model_dict['learner'] = 'catboost'
            surrogate_model_tag_real += '_' + surrogate_model_dict['learner']
            surrogate_model_dict['src_use'] = transfer_analysis and N_SRC_DOMAIN
            if surrogate_model_dict['src_use']:
                surrogate_model_tag_real += '_su'
        case "our":
            surrogate_model_dict['v'] = 1
            surrogate_model_dict['learner'] = 'GBRT'
            #surrogate_model_dict['learner'] = 'XGBoost'
            #surrogate_model_dict['learner'] = "catboost"
            surrogate_model_tag_real += '_' + surrogate_model_dict['learner']
            #surrogate_model_dict['meta_learner'] = surrogate_model_dict['learner']
            #surrogate_model_tag_real += '_meta-' + surrogate_model_dict['learner']
            surrogate_model_dict['domain_loss'] = ''
        case "our2":
            surrogate_model_dict['v'] = 3

            #surrogate_model_dict['learner'] = 'M5P'
            #surrogate_model_dict['learner'] = 'sklearnmlp'
            #surrogate_model_dict['learner'] = 'tranmlp'
            surrogate_model_dict['learner'] = 'GBRT'
            #surrogate_model_dict['learner'] = 'XGBoost'
            #surrogate_model_dict['learner'] = 'catboost'
            surrogate_model_tag_real += '_' + surrogate_model_dict['learner']

            surrogate_model_dict['domain_loss'] = ''
            if 'tranmlp' == surrogate_model_dict['learner']:
                #surrogate_model_dict['domain_loss'] = 'mkmmd_2'
                surrogate_model_tag_real += '_' + surrogate_model_dict['domain_loss']
        case "our3":
            surrogate_model_dict['v'] = 1
            surrogate_model_dict['learner'] = 'GBRT'
            #surrogate_model_dict['learner'] = 'XGBoost'
            #surrogate_model_dict['learner'] = "catboost"
            surrogate_model_tag_real += '_' + surrogate_model_dict['learner']
            surrogate_model_dict['meta_learner'] = surrogate_model_dict['learner']
            surrogate_model_tag_real += '_meta-' + surrogate_model_dict['learner']
            surrogate_model_dict['domain_loss'] = 'mkmmd_2'
            surrogate_model_tag_real += '_' + surrogate_model_dict['domain_loss']
        case "our4_base_lw":
            surrogate_model_dict['v'] = 9
            surrogate_model_dict['n_init_samples'] = N_SAMPLES_INIT
            surrogate_model_tag_real += '_i' + str(surrogate_model_dict['n_init_samples'])
            surrogate_model_dict['meta_learner'] = ''
            surrogate_model_dict['sample_weight'] = ''
            surrogate_model_dict['reweight-iter'] = False
            surrogate_model_dict['learner_weight'] = 4
            surrogate_model_dict['domain_weight'] = False
        case "our4_hbo_lw":
            surrogate_model_dict['v'] = 9
            surrogate_model_dict['n_init_samples'] = 10
            surrogate_model_tag_real += '_i' + str(surrogate_model_dict['n_init_samples'])
            surrogate_model_dict['meta_learner'] = ''
            surrogate_model_dict['sample_weight'] = ''
            surrogate_model_dict['reweight-iter'] = False
            surrogate_model_dict['learner_weight'] = 4
            surrogate_model_dict['domain_weight'] = False
        case "our4_hbo_lw_sdv4":
            surrogate_model_dict['v'] = 9
            surrogate_model_dict['n_init_samples'] = 10
            surrogate_model_tag_real += '_i' + str(surrogate_model_dict['n_init_samples'])
            surrogate_model_dict['meta_learner'] = ''
            surrogate_model_dict['sample_weight'] = 'sdv4'
            surrogate_model_dict['reweight-iter'] = False
            surrogate_model_dict['learner_weight'] = 4
            surrogate_model_dict['domain_weight'] = False
        case "our4_lw_sdv3":
            #surrogate_model_dict['v'] = 9
            surrogate_model_dict['v'] = 11  # SVR
            surrogate_model_dict['n_init_samples'] = N_SAMPLES_INIT
            surrogate_model_tag_real += '_i' + str(surrogate_model_dict['n_init_samples'])
            surrogate_model_dict['meta_learner'] = ''
            surrogate_model_dict['sample_weight'] = 'sdv3'
            surrogate_model_tag_real += '_sw2-' + surrogate_model_dict['sample_weight']
            surrogate_model_dict['reweight-iter'] = True
            surrogate_model_dict['reweight-beta'] = 'Exp'
            surrogate_model_dict['learner_weight'] = 0
            surrogate_model_dict['domain_weight'] = True
            surrogate_model_tag_real += '_dw2'
        case "our4_lw_sdv4":
            #surrogate_model_dict['v'] = 9
            #surrogate_model_dict['v'] = 11  # SVR
            surrogate_model_dict['v'] = 13  # AdaMLP
            surrogate_model_dict['n_init_samples'] = N_SAMPLES_INIT
            surrogate_model_tag_real += '_i' + str(surrogate_model_dict['n_init_samples'])
            surrogate_model_dict['meta_learner'] = ''
            surrogate_model_dict['sample_weight'] = 'sdv4'
            surrogate_model_tag_real += '_sw2-' + surrogate_model_dict['sample_weight']
            surrogate_model_dict['reweight-iter'] = False
            surrogate_model_dict['reweight-beta'] = ''
            surrogate_model_dict['learner_weight'] = 0
            surrogate_model_dict['domain_weight'] = True
            surrogate_model_tag_real += '_dw2'
        case "our4":
            #surrogate_model_dict['v'] = 9
            surrogate_model_dict['v'] = 11 #SVR
            #surrogate_model_dict['v'] = 12 #MLP
            surrogate_model_dict['n_init_samples'] = N_SAMPLES_INIT
            surrogate_model_tag_real += '_i' + str(surrogate_model_dict['n_init_samples'])
            surrogate_model_dict['meta_learner'] = ''
            #surrogate_model_dict['meta_learner'] = 'SVR'
            #surrogate_model_dict['meta_learner'] = 'XGBoost'
            if '' != surrogate_model_dict['meta_learner']:
                surrogate_model_tag_real += '_meta' + surrogate_model_dict['meta_learner']

            #surrogate_model_dict['sample_weight'] = ''
            #surrogate_model_dict['sample_weight'] = 'unit'
            #surrogate_model_dict['sample_weight'] = 'sdv2'
            #surrogate_model_dict['sample_weight'] = 'sdv4'
            surrogate_model_dict['sample_weight'] = 'sdv3'
            surrogate_model_dict['reweight-iter'] = False
            surrogate_model_dict['reweight-beta'] = ''
            if '' != surrogate_model_dict['sample_weight']:
                surrogate_model_tag_real += '_sw2-' + surrogate_model_dict['sample_weight']
            if 'sdv3' == surrogate_model_dict['sample_weight']:
                surrogate_model_dict['reweight-beta'] = 'Exp'
                #surrogate_model_dict['reweight-beta'] = 'Ada'
                #surrogate_model_dict['reweight-beta'] = 'v5'
                surrogate_model_dict['reweight-iter'] = True
            if '' != surrogate_model_dict['reweight-beta']:
                surrogate_model_tag_real += '_beta-' + surrogate_model_dict['reweight-beta']
            #surrogate_model_dict['learner_weight'] = False
            surrogate_model_dict['learner_weight'] = 4
            if surrogate_model_dict['learner_weight']:
                surrogate_model_tag_real += '_lwv' + str(surrogate_model_dict['learner_weight'])

            #surrogate_model_dict['domain_weight'] = False
            surrogate_model_dict['domain_weight'] = True
            if surrogate_model_dict['domain_weight']:
                surrogate_model_tag_real += '_dw2'
            #surrogate_model_dict['learner'] = 'GBRT'
            #surrogate_model_dict['learner'] = 'XGBoost'
            #surrogate_model_dict['learner'] = "catboost"
            #surrogate_model_tag_real += '_' + surrogate_model_dict['learner']
            #surrogate_model_dict['meta_learner'] = surrogate_model_dict['learner']
            #surrogate_model_tag_real += '_meta-' + surrogate_model_dict['learner']
            #surrogate_model_dict['domain_loss'] = 'mkmmd_2'
            #surrogate_model_tag_real += '_' + surrogate_model_dict['domain_loss']
        case _:
            print(f"no def surrogate_model_tag={surrogate_model_tag}")
            exit(1)
    if '' != surrogate_model_dict['y_transform']:
        surrogate_model_tag_real += '_' + surrogate_model_dict['y_transform']
    surrogate_model_tag_real += "_vv" + str(surrogate_model_dict['v'])
    if N_SRC_DOMAIN_TRAIN:
        surrogate_model_tag_real += '_s' + str(N_SRC_DOMAIN)
        surrogate_model_tag_real += '_sn' + str(N_SRC_DOMAIN_TRAIN)
    surrogate_model_tag_real += '_dn' + str(N_SAMPLES_INIT)

    surrogate_model_dict['tag'] = surrogate_model_tag_real
    return surrogate_model_dict

#result_array = np.zeros(n_experiment)
result_time_array = np.zeros(n_experiment)
result_npccs_array = np.zeros(n_experiment)
result_mae_array = np.zeros(n_experiment)
result_mape_array = np.zeros(n_experiment)
result_distance_value = np.zeros(n_experiment)

def reset_result():
    #result_array = np.zeros(n_experiment)
    result_time_array = np.zeros(n_experiment)
    result_npccs_array = np.zeros(n_experiment)
    result_mae_array = np.zeros(n_experiment)
    result_mape_array = np.zeros(n_experiment)
    result_distance_value = np.zeros(n_experiment)


def save_result(surrogate_model_tag_real, case_name_src):
    result_summary_file = open("log_summary_" + metric_name + "/" + case_name_src + '-s' + str(N_SRC_DOMAIN) + "-summary.txt", "a")
    result_summary_file.write(
        "%-60s %-10f %-10f %-10f %-10f %-10f %-10f %-10f %-10f %-2d %-15s \n" \
        % (surrogate_model_tag_real,
           #result_array.mean(), ci(result_array),
           result_npccs_array.mean(), ci(result_npccs_array),
           result_mae_array.mean(), ci(result_mae_array),
           result_mape_array.mean(), ci(result_mape_array),
           result_time_array.mean(), ci(result_time_array),
           len(result_npccs_array), hostname,
           )
    )
    result_summary_file.close()


def save_result_tranfer(surrogate_model_tag_real, case_name_config, case_name_src, distance_value):
    result_summary_file = open("log_summary_transfer_" + metric_name + "/" + case_name_config + '-s' + str(N_SRC_DOMAIN) + "-summary.txt", "a")
    result_summary_file.write(
        "%-60s %-10f %-10f %-10f %-10f %-10f %-10f %-10f %-10f %-2d %-15s %-10f %-15s \n" \
        % (surrogate_model_tag_real,
           #result_array.mean(), ci(result_array),
           result_npccs_array.mean(), ci(result_npccs_array),
           result_mae_array.mean(), ci(result_mae_array),
           result_mape_array.mean(), ci(result_mape_array),
           result_time_array.mean(), ci(result_time_array),
           len(result_npccs_array), hostname,
           distance_value,
           case_name_src,
           )
    )
    result_summary_file.close()

if __name__ == '__main__':
    if transfer_analysis:
        if N_SRC_DOMAIN:
            if case_range is not None:
                case_name_to_run_list = case_names[case_range*3:max(case_range*3+3,len(case_names))]
            else:
                case_name_to_run_list = case_names
        else:
            case_name_to_run_list = [case_name_config]
    else:
        case_name_to_run_list = case_names
        #case_name_to_run_list = [case_name_config]
    for case_name_src in case_name_to_run_list:
        if transfer_analysis:
            case_name = case_name_config
        else:
            case_name = case_name_src
        for surrogate_model_tag in surrogate_model_tag_list:
            surrogate_model_config = get_surrogate_model(surrogate_model_tag)
            surrogate_model_tag_real = surrogate_model_config['tag']
            metrics_all = read_metrics_cases('data_all_simpoint/')
            experiment_range = range(n_experiment)
            if exp_id is not None:
                experiment_range = range(exp_id, exp_id + 1)
            reset_result()
            for exp_i in experiment_range:
                random.seed(exp_i)
                if transfer_analysis:
                    # only for 1 src domain
                    src_domain_list = [case_name_src] #get_src_domain_list(case_name, random_seed=1)
                    if case_name_src == case_name:
                        src_domain_list = []
                        if N_SRC_DOMAIN:
                            continue
                else:
                    src_domain_list = get_src_domain_list(case_name, random_seed=exp_i)
                    #src_domain_list = ["523.1-refrate-1"]
                result_filename_prefix = "log/" + metric_name + '-' + case_name + "-" + surrogate_model_tag_real + "-exp-" + str(exp_i)
                print(f"result_filename_prefix={result_filename_prefix}")
                #if print_info:
                print(f"src_domain_list= {src_domain_list}")
                startTime = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
                log_file = open(result_filename_prefix + ".log", "w")
                train_data, X_test, Y_test_real = gen_train_data(case_name=case_name, metrics_all=metrics_all, src_domain_list=src_domain_list, surrogate_model_tag=surrogate_model_tag, random_seed=exp_i)
                distance_value_exp_i = None
                match surrogate_model_tag:
                    case 'PACT07':
                        from model_PACT07 import model_PACT07
                        Y_test_pred = model_PACT07(train_data, X_test, metrics_all[case_name])
                    case 'ICCD09':
                        from model_PACT07 import model_ICCD09
                        Y_test_pred = model_ICCD09(src_domain_list, train_data, X_test)
                    case 'M5P':
                        from model_M5P import model_M5P
                        Y_test_pred = model_M5P(train_data, X_test)
                    case 'EAC':
                        from model_EAC import model_EAC
                        Y_test_pred = model_EAC(train_data, X_test)
                    case 'TrDSE':
                        from model_TrDSE import model_TrDSE
                        Y_test_pred = model_TrDSE(train_data, X_test, surrogate_model_config, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'TrEE':
                        from model_TrEE import model_TrEE
                        Y_test_pred = model_TrEE(train_data, X_test)
                    case 'MPGM':
                         model = model_MPGM(surrogate_model_config, metrics_all, src_domain_list, train_data, X_test,
                                            n=exp_i, log_file=log_file, print_info=print_info)
                         Y_test_pred = model.predict(X_test)
                         #print(f"Y_test_pred range={np.max(Y_test_pred)}")
                    case 'CASH':
                        from model_CASH import model_CASH
                        Y_test_pred = model_CASH(train_data, X_test)
                    case 'our':
                        from model_our import model_our
                        Y_test_pred = model_our(train_data, X_test, surrogate_model_config, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'our2':
                        from model_our2 import model_our2
                        Y_test_pred = model_our2(train_data, X_test, surrogate_model_config, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'our3':
                        from model_our3 import model_our3
                        Y_test_pred = model_our3(train_data, X_test, surrogate_model_config, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'our4_base_lw':
                        from model_our4 import model_our4
                        Y_test_pred = model_our4(train_data, X_test, surrogate_model_config, metrics_all=metrics_all[case_name], src_domain_list=src_domain_list, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'our4_hbo_lw':
                        from model_our4 import model_our4
                        Y_test_pred = model_our4(train_data, X_test, surrogate_model_config, metrics_all=metrics_all[case_name], src_domain_list=src_domain_list, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'our4_hbo_lw_sdv4':
                        from model_our4 import model_our4
                        Y_test_pred = model_our4(train_data, X_test, surrogate_model_config, metrics_all=metrics_all[case_name], src_domain_list=src_domain_list, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'our4_lw_sdv3':
                        from model_our4 import model_our4
                        Y_test_pred = model_our4(train_data, X_test, surrogate_model_config, metrics_all=metrics_all[case_name], src_domain_list=src_domain_list, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'our4_lw_sdv4':
                        from model_our4 import model_our4
                        Y_test_pred = model_our4(train_data, X_test, surrogate_model_config, metrics_all=metrics_all[case_name],
                                                 src_domain_list=src_domain_list, log_file=log_file, print_info=print_info,
                                                 Y_real=Y_test_real)
                    case 'our4':
                        from model_our4 import model_our4
                        Y_test_pred = model_our4(train_data, X_test, surrogate_model_config, metrics_all=metrics_all[case_name], src_domain_list=src_domain_list, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case 'program_specific':
                        from model_program_specific import model_program_specific
                        Y_test_pred, distance_value_exp_i = model_program_specific(train_data, X_test, surrogate_model_config, metric_name, log_file=log_file, print_info=print_info, Y_real=Y_test_real)
                    case _:
                        print(f"no def model {surrogate_model_config['model_name']}")
                        exit(1)
                npccs, mae, mape = evaluate(Y_real=Y_test_pred, Y_pred=Y_test_real)
                time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                              "%Y-%m-%d %H:%M:%S") - startTime
                SRC_DOMAIN_ENCODE_LIST_STR = get_SRC_DOMAIN_ENCODE_LIST_STR(src_domain_list)
                log_file.write(f"SRC_DOMAIN_ENCODE_LIST_STR={SRC_DOMAIN_ENCODE_LIST_STR} \n")
                log_file.write(f"SRC_DOMAIN_LIST={src_domain_list} \n")
                log_file.write(f"time_used={time_used} \n")
                log_file.write(f"npccs={npccs} \n")
                log_file.write(f"mae={mae} \n")
                log_file.write(f"mape={mape} \n")
                print(f"{result_filename_prefix} = npccs={npccs}, mae={mae}, mape={mape}")
                result_npccs_array[exp_i] = npccs
                result_mae_array[exp_i] = mae
                result_mape_array[exp_i] = mape
                result_time_array[exp_i] = time_used.total_seconds()
                if distance_value_exp_i is not None:
                    result_distance_value[exp_i] = np.mean(distance_value_exp_i)
                log_file.close()

            # end after all experiments
            if transfer_analysis:
                if 0:
                    from program_inherent_similarity import get_domain_distance
                    distance_value = get_domain_distance(domain_feature_0, domain_feature_1, domain_loss='wasserstein').item()
                elif 1:
                    from program_inherent_similarity import get_all_distance_from_file
                    distance_data, data_config_str = get_all_distance_from_file(metric_iter=metric_name, domain_loss='wasserstein')
                    for domain_iter, domain_case_name_iter in enumerate(src_domain_list):
                        domain_weight_unique = distance_data[get_domain_id(case_name), get_domain_id(domain_case_name_iter)]
                    if len(src_domain_list):
                        distance_value = np.mean(domain_weight_unique)
                    else:
                        distance_value = 0
                else:
                    distance_value = np.mean(result_distance_value)
                save_result_tranfer(surrogate_model_tag_real=surrogate_model_tag_real, case_name_config=case_name, case_name_src=case_name_src, distance_value=distance_value)
            else:
                save_result(surrogate_model_tag_real=surrogate_model_tag_real, case_name_src=case_name)
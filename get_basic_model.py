from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from MLP_Predictor_Transfer import MLP_Predictor_Transfer
from model_tree.models.linear_regr import linear_regr
from model_tree.src.ModelTree import ModelTree
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, \
    ExtraTreesRegressor
from sklearn.linear_model import LinearRegression


def get_basic_model(learner, metric_name, return_config=False):
    #ada_params_cpi = {'n_estimators': 20, 'learning_rate': 0.001}
    sample_weight_ability = False
    if 'linear' == learner:
        base_estimator = LinearRegression()
    elif 'M5P' == learner:
        param_cpi = {'max_depth': 10, 'min_samples_leaf': 9}
        base_estimator = ModelTree(linear_regr(), search_type="grid", n_search_grid=3, **param_cpi)
        #sample_weight_ability = False
        sample_weight_ability = True
    elif 'SVR' == learner:
        #svr_params_cpi = {'kernel': 'rbf', }
        #svr_params_cpi = {'degree': 1, 'kernel__k2__length_scale': 2, 'kernel__k2__nu': 1.5}
        svr_params_power = {'degree': 8}
        base_estimator = SVR(**svr_params_power, kernel=1.0 * Matern(length_scale=3.0, length_scale_bounds=(1e-1, 20.0), nu=2.5))
        sample_weight_ability = True
    elif 'GP' == learner:
        gp_params_cpi = {'n_restarts_optimizer': 10} #, 'kernel__k2__nu': 2.5, 'normalize_y': False}
        base_estimator = GaussianProcessRegressor(kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 20.0), nu=2.5), **gp_params_cpi)
    elif 'RF' == learner:
        HBO_params_cpi = {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 1}
        HBO_params_power = {'n_estimators': 199, 'max_depth': 30, 'min_samples_leaf': 1}
        HBO_params = HBO_params_cpi #if metric_name == 'CPI' else HBO_params_power
        base_estimator = RandomForestRegressor(**HBO_params)
        sample_weight_ability = True
    elif "ET" == learner:
        HBO_params_cpi = {'n_estimators': 200, 'max_depth': 5, 'min_samples_leaf': 1, 'max_features': 1.0}
        HBO_params_power = {'n_estimators': 182, 'max_depth': 30, 'min_samples_leaf': 1, 'max_features': 1.0}
        HBO_params = HBO_params_cpi #if metric_name == 'CPI' else HBO_params_power
        base_estimator = ExtraTreesRegressor(**HBO_params)
        sample_weight_ability = True
    elif 'GBRT' == learner:
        #HBO_params_cpi = {'n_estimators': 198, 'learning_rate': 0.1, 'max_depth': 12, 'subsample': 0.5}
        HBO_params_cpi = {'n_estimators': 184, 'learning_rate': 0.1, 'max_depth': 2, 'subsample': 0.8}
        HBO_params_power = {'n_estimators': 299, 'learning_rate': 0.2, 'max_depth': 3, 'subsample': 0.8}
        HBO_params = HBO_params_cpi# if metric_name == 'CPI' else HBO_params_power
        base_estimator = GradientBoostingRegressor(**HBO_params)
        sample_weight_ability = True
    elif 'AdaGBRT' == learner:
        HBO_params_cpi = {'n_estimators': 198, 'learning_rate': 0.1, 'max_depth': 12, 'subsample': 0.5}
        base_estimator_ = GradientBoostingRegressor(**HBO_params_cpi)
        HBO_params_cpi_ada = {'n_estimators': 198, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(base_estimator=base_estimator_, **HBO_params_cpi_ada)
        sample_weight_ability = True
    elif 'AdaMLP' == learner:
        base_estimator_ = MLPRegressor(hidden_layer_sizes=(10), max_iter=10000, solver='adam')
        HBO_params_cpi_ada = {'n_estimators': 198, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(base_estimator=base_estimator_, **HBO_params_cpi_ada)
        sample_weight_ability = True        
    elif "XGBoost" == learner:
        from xgboost import XGBRegressor
        HBO_params_cpi = {'n_estimators': 174, 'learning_rate': 0.15, 'max_depth': 5, 'booster': 'dart', 'subsample': 0.8}
        HBO_params_power = {'n_estimators': 114, 'learning_rate': 0.15, 'max_depth': 6, 'booster': 'dart', 'subsample': 0.8}
        base_estimator = XGBRegressor(**HBO_params_power, n_jobs=2, nthread=None, )
        sample_weight_ability = False
    elif 'catboost' == learner:
        HBO_params_cpi = {'n_estimators': 155, 'depth': 5, 'subsample': 1.0, 'early_stopping_rounds': 144,
                          'grow_policy': 'Lossguide'}
        from catboost import CatBoostRegressor
        base_estimator = CatBoostRegressor(**HBO_params_cpi, silent=True)  # , thread_count=-1) verbose=False,
        # base_estimator_ada = AdaBoostRegressor(base_estimator=meta_base_estimator_2, **ada_params_cpi, )
    elif 'sklearnmlp' == learner:
        base_estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10000, solver='lbfgs')
    elif 'sklearnmlp10' == learner:
        base_estimator = MLPRegressor(hidden_layer_sizes=(10), max_iter=10000, solver='adam')
    '''
    else:
        meta_base_estimator = MLP_Predictor_Transfer(in_channel=8, out_channel=1, drop_rate=0.2,
                                                  use_bias=True, use_drop=True, initial_lr=0.001,
                                                  # n_src_samples=N_SRC_DOMAIN_TRAIN * N_SRC_DOMAIN,
                                                  domain_loss_mode=False,
                                                  domain_loss=surrogate_model_config['domain_loss'],
                                                  print_info=False)
    '''
    if return_config:
        return base_estimator, sample_weight_ability
    else:
        return base_estimator
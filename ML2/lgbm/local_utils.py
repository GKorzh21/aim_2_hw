import numpy as np
from time import time as tm
from tqdm import tqdm
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import optuna
from optuna.visualization import *
from functools import partial
import gc

def make_subsample_test():
    N = int(1e7)
    n_features = 50
    X_dummy, y_dummy = make_classification(N, n_features, n_informative=30, random_state=911)

    sizes = [50_000, 200_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    time_goss = []
    time_goss_bad = []
    time_plain_subsample = []
    time_plain = []
    
    
    n_trees = 100
    
    for size in tqdm(sizes):
        params = {
            'nthread': 32,
            'verbose': -1,
            'device_type': 'cpu',
            'seed': 911
        }
        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        data.construct()
        
        params.update({'data_sample_strategy': 'goss', 'top_rate': 0.001, 'other_rate': 1e-7})
        #params.update({'data_sample_strategy': 'goss', 'top_rate': 0.1, 'other_rate': 0.5})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_goss.append(tm() - stm)
        
        
        params.update({'top_rate': 0.999, 'other_rate': 0.001})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_goss_bad.append(tm() - stm)
        
        params.update({'data_sample_strategy': 'bagging', 'subsample_freq': 1, 'subsample': 0.5})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_plain_subsample.append(tm() - stm)
        
        
        params.update({'subsample_freq': 0, 'subsample': 1.})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_plain.append(tm() - stm)

    

    plt.plot(sizes, time_goss, label='goss')
    plt.plot(sizes, time_goss_bad, label='goss_bad')
    plt.plot(sizes, time_plain_subsample, label='plain subsample')
    plt.plot(sizes, time_plain, label='plain')
    plt.legend(fontsize=15)
    plt.gca().tick_params('both', labelsize=15)
    plt.ylabel('train time, sec', fontsize=15)
    plt.xlabel('data size', fontsize=15)
    plt.xscale('log')
    plt.gcf().set_size_inches(8, 5)
    #return sizes, time_goss, time_goss_bad, time_plain_subsample, time_plain


def make_quantized_grad_test():
    N = int(1e7)
    n_features = 10
    #X_dummy, y_dummy = make_classification(N, n_features, n_informative=2, random_state=911)
    X_dummy, y_dummy = make_classification(N, n_features, n_informative=10, random_state=911)

    sizes = [50_000, 200_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    time_quant = []
    time_plain = []
    
    #n_trees = 10
    n_trees = 100
    
    for size in tqdm(sizes):
        params = {
            'nthread': 1,
            'verbose': -1,
            'device_type': 'cpu',
            'seed': 911,

            'max_leaves': 128
        }
        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        data.construct()
        
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_plain.append(tm() - stm)
        
        #params.update({'use_quantized_grad': True, 'quant_train_renew_leaf': False})
        params.update({'use_quantized_grad': True, 'quant_train_renew_leaf': True})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_quant.append(tm() - stm)
        
    

    plt.plot(sizes, time_quant, label='quantized_grad')
    plt.plot(sizes, time_plain, label='plain')
    plt.legend(fontsize=15)
    plt.gca().tick_params('both', labelsize=15)
    plt.ylabel('train time, sec', fontsize=15)
    plt.xlabel('data size', fontsize=15)
    plt.xscale('log')
    plt.gcf().set_size_inches(8, 5)


def make_linear_tree_test():
    N = int(1e7)
    n_features = 10
    
    X_dummy, y_dummy = make_classification(N, n_features, n_informative=2, random_state=911)

    sizes = [50_000, 200_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    time_linear = []
    time_plain = []
    
    
    n_trees = 10
    
    for size in tqdm(sizes):
        params = {
            'nthread': 32,
            'verbose': -1,
            'device_type': 'cpu',
            'seed': 911,

            'max_leaves': 128
        }
        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_plain.append(tm() - stm)

        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        params.update({'linear_tree': True})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_linear.append(tm() - stm)
        
    

    plt.plot(sizes, time_linear, label='linear_tree')
    plt.plot(sizes, time_plain, label='plain')
    plt.legend(fontsize=15)
    plt.gca().tick_params('both', labelsize=15)
    plt.ylabel('train time, sec', fontsize=15)
    plt.xlabel('data size', fontsize=15)
    plt.xscale('log')
    plt.gcf().set_size_inches(8, 5)


def make_dart_test():
    N = int(1e7)
    n_features = 10
    X_dummy, y_dummy = make_classification(N, n_features, n_informative=2, random_state=911)

    sizes = [50_000, 200_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    time_dart = []
    time_plain = []
    
    
    n_trees = 100
    
    for size in tqdm(sizes):
        params = {
            'nthread': 32,
            'verbose': -1,
            'device_type': 'cpu',
            'seed': 911,

            'max_leaves': 128
        }
        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_plain.append(tm() - stm)

        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        params.update({'boosting': 'dart'})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_dart.append(tm() - stm)
        
    

    plt.plot(sizes, time_dart, label='dart')
    plt.plot(sizes, time_plain, label='plain')
    plt.legend(fontsize=15)
    plt.gca().tick_params('both', labelsize=15)
    plt.ylabel('train time, sec', fontsize=15)
    plt.xlabel('data size', fontsize=15)
    plt.xscale('log')
    plt.gcf().set_size_inches(8, 5)


def make_eta_test():
    def objective(trial, X_tr, y_tr):
        params = {
            'objective': 'binary',
            'nthread': 32,
            'verbose': -1,

            'eta': trial.suggest_float('eta', 1e-7, 10, log=True)
        }

        tr_lgb = lgb.Dataset(X_tr, y_tr)
        stm = tm()
        model = lgb.train(params, tr_lgb, num_boost_round=100)
        return tm() - stm

    X_dummy, y_dummy = make_classification(400_000, 20, n_informative=2, random_state=911)
    
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler())
    func = partial(objective, X_tr=X_dummy, y_tr=y_dummy)
    study.optimize(func, n_trials=300, n_jobs=4)
    return study



def make_subsample_test_fair(n_t):
    N = int(1e7)
    n_features = 50
    X_dummy, y_dummy = make_classification(N, n_features, n_informative=30, random_state=911)

    sizes = [50_000, 200_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    time_goss = []
    time_plain = []
    
    n_trees = n_t
    
    for size in tqdm(sizes):
        params = {
            'nthread': 1,
            'verbose': -1,
            'device_type': 'cpu',
            'seed': 911,
            'enable_cache': False
        }
        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        data.construct()
        
        # GOSS
        gc.collect()
        params.update({'data_sample_strategy': 'goss', 'top_rate': 0.1, 'other_rate': 0.2})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_goss.append(tm() - stm)
        
        # Plain
        gc.collect()
        params.update({'data_sample_strategy': 'bagging', 'subsample_freq': 0, 'subsample': 1.})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_plain.append(tm() - stm)

    return (time_plain[-1], time_goss[-1])

    plt.plot(sizes, time_goss, label='goss')
    plt.plot(sizes, time_plain, label='plain')
    plt.legend(fontsize=15)
    plt.gca().tick_params('both', labelsize=15)
    plt.ylabel('train time, sec', fontsize=15)
    plt.xlabel('data size', fontsize=15)
    plt.xscale('log')
    plt.title(f"n_trees = {n_trees}")
    plt.gcf().set_size_inches(8, 5)



def make_quantized_grad_test_fair(n_t, max_leaves):
    N = int(1e7)
    n_features = 10
    X_dummy, y_dummy = make_classification(N, n_features, n_informative=8, random_state=911)

    sizes = [50_000, 200_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    time_quant = []
    time_plain = []
    
    n_trees = n_t
    
    for size in tqdm(sizes):
        params = {
            'nthread': 1,
            'verbose': -1,
            'device_type': 'cpu',
            'seed': 911,

            'max_leaves': max_leaves
        }
        data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size])
        data.construct()
        
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_plain.append(tm() - stm)
        
        params.update({'use_quantized_grad': True, 'quant_train_renew_leaf': True})
        stm = tm()
        model = lgb.train(params, data, num_boost_round=n_trees)
        time_quant.append(tm() - stm)
        
    return (time_plain[-1], time_quant[-1])

    plt.plot(sizes, time_quant, label='quantized_grad')
    plt.plot(sizes, time_plain, label='plain')
    plt.legend(fontsize=15)
    plt.gca().tick_params('both', labelsize=15)
    plt.ylabel('train time, sec', fontsize=15)
    plt.xlabel('data size', fontsize=15)
    plt.xscale('log')
    plt.gcf().set_size_inches(8, 5)
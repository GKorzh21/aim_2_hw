import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb, catboost as cb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import polars as pl
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def pretty_print(iterable, container='list', max_symbols=100, as_strings=None, return_result=False, prefix='', suffix=''):
    '''
    Рисует (и при надобности возвращает) iterable -> s = '[el for el in iterable]' в красивом виде!
    
    container: 'list' -> [...], 'set' -> {...}, 'tuple' -> (...)
    max_symbols: макс. желаемое количество символов в одной строке
    as_strings: оборачивать ли элементы iterable в одинарные кавычки
    '''
    brackets = {'list': '[]', 'set': '{}', 'tuple': '()'}
    n_symbol = 1
    s = ''
    for el in iterable:
        is_str = as_strings if as_strings is not None else isinstance(el, str)
        if pd.isna(el):
            if isinstance(el, float):
                el = 'np.nan'
            else:
                el = 'None'
            is_str = False

        el = prefix + str(el) + suffix
        tmp = f"'{el}', " if is_str else f"{el}, "
        tmp_len = len(tmp)
        if n_symbol + tmp_len > max_symbols:
            s += '\n\t'
            n_symbol = 4
        s += tmp
        n_symbol += tmp_len
    s = s[:-2]
    res = brackets[container][0] + s + brackets[container][1]
    if return_result:
        return res
    print(res)


def extract_attr_by_leaf_matrix(t, leaf_matrix, attr_name):
    t_prep = t.copy()
    t_prep['leaf_id'] = t_prep.node_index.str.split('-').str.get(1).str.slice(1, 10).astype(int)
    t_prep = t_prep.query('split_gain.isnull()').pivot_table(values=attr_name, index='tree_index', columns='leaf_id')
    res = []
    for i in range(t_prep.shape[0]): # кол-во деревьев
        res.append(t_prep.iloc[i][leaf_matrix[:, i]])
    res = np.array(res).T
    return res


def plot_feature_info(t, top_k=20, return_activations=False):
    t_prep = t.copy()
    t_prep.split_feature = t_prep.split_feature.astype('category')
    t_prep['log_norm_gain'] = np.log2(np.log(1 + t_prep.split_gain / t_prep['count'] ** 2))
    t_prep['tree_total_gain'] = t_prep.groupby(['tree_index'], observed=False)['log_norm_gain'].transform('sum')
    t_prep['activations'] = t_prep.log_norm_gain / (t_prep.tree_total_gain + 1e-8)
    t_prep = (
        t_prep.query('~split_feature.isnull()')
        .groupby(['tree_index', 'split_feature'], observed=False)['activations']
        .sum()
        .to_frame('activations')
        .reset_index()
    )
    t_prep = t_prep.pivot_table(values='activations', columns='tree_index', index='split_feature')
    # median_cnt = t_prep.max(axis=1)
    # idx = np.argsort(median_cnt)[::-1][:top_k]
    
    sns.clustermap(t_prep, cmap='coolwarm', col_cluster=False, robust=False)
    plt.gcf().set_size_inches(11, top_k/6 + 1)
    if return_activations:
        return t_prep


def plot_feature_depth(t, top_k=20):
    t_prep = t.copy()
    t_prep.split_feature = t_prep.split_feature.astype('category')
    t_prep = (
        t_prep.query('~split_feature.isnull()')
        .groupby(['node_depth', 'split_feature'], observed=True)
        .size()
        .to_frame('cnt')
        .reset_index()
    )
    t_prep.cnt = t_prep.cnt / 2**(t_prep.node_depth - 1)
    t_prep = t_prep.pivot_table(values='cnt', columns='node_depth', index='split_feature')
    median_cnt = t_prep.sum(axis=1)
    idx = np.argsort(median_cnt)[::-1][:top_k]
    
    sns.heatmap(t_prep.iloc[idx], cmap='coolwarm', annot=False, robust=True)
    plt.gcf().set_size_inches(6, top_k/6 + 1)


def plot_ensemble_profile(t):
    t['abs_value'] = t.value.abs()
    mosaic = [
        ['weights', 'abs_weights', 'depth'],
        ['cnt', 'hess', 'gain']
    ]
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(15, 8))

    ax['weights'].set_title('Веса в листьях', fontsize=12)
    sns.scatterplot(t.query('split_gain.isnull()'), x='tree_index', y='value', s=50, ax=ax['weights'])
    sns.lineplot(t.query('split_gain.isnull()'), x='tree_index', y='value',  color='orange', lw=3, ax=ax['weights'])

    ax['abs_weights'].set_title('Модули весов в листьях', fontsize=12)
    sns.scatterplot(t.query('split_gain.isnull()'), x='tree_index', y='abs_value', s=50, ax=ax['abs_weights'])
    sns.lineplot(t.query('split_gain.isnull()'), x='tree_index', y='abs_value',  color='orange', lw=3, ax=ax['abs_weights'])

    ax['cnt'].set_title('Количество объектов в листьях', fontsize=12)
    sns.scatterplot(t.query('split_gain.isnull()'), x='tree_index', y='count', s=50, ax=ax['cnt'])
    sns.lineplot(t.query('split_gain.isnull()'), x='tree_index', y='count',  color='orange', lw=3, ax=ax['cnt'])
    ax['cnt'].set_yscale('log')

    ax['hess'].set_title('sum_hessian в знаменателе при сплитах', fontsize=12)
    sns.scatterplot(t, x='tree_index', y='weight', s=50, ax=ax['hess'])
    sns.lineplot(t, x='tree_index', y='weight',  color='orange', lw=3, ax=ax['hess'])
    ax['hess'].set_yscale('log')

    ax['depth'].set_title('Глубина деревьев', fontsize=12)
    sns.scatterplot(t, x='tree_index', y='node_depth', s=50, ax=ax['depth'])
    sns.lineplot(t, x='tree_index', y='node_depth',  color='orange', lw=3, ax=ax['depth'])

    ax['gain'].set_title('Gain', fontsize=12)
    sns.scatterplot(t, x='tree_index', y='split_gain', s=50, ax=ax['gain'])
    sns.lineplot(t, x='tree_index', y='split_gain',  color='orange', lw=3, ax=ax['gain'])
    ax['gain'].set_yscale('log')

    fig.tight_layout()


def plot_lgbm_importance(model, features, importance_type='split', top_k=20, sklearn_style=False, imps=None, round_to=0):
    if sklearn_style and imps is None:
        imps = model.feature_importances_
    elif imps is None:
        imps = model.feature_importance(importance_type=importance_type)
        
    idx = np.argsort(imps)
    sorted_imps = imps[idx][::-1][:top_k][::-1]
    sorted_features = np.array(features)[idx][::-1][:top_k][::-1]
    if round_to == 0:
        sorted_imps = sorted_imps.astype(int)
    else:
        sorted_imps = np.round(sorted_imps, round_to)
        
    bar_container = plt.barh(width=sorted_imps, y=sorted_features)
    plt.bar_label(bar_container, sorted_imps, color='red')
    plt.gcf().set_size_inches(5, top_k/6 + 1)
    plt.xlabel(importance_type, fontsize=15)
    sns.despine()


def get_shadow_features(tr, val, n_float=5, n_cat_big=5, n_cat_small=5):
    col_names = [f'shadow_float_{i+1}' for i in range(5)]
    tr_shadow = pd.DataFrame(np.random.randn(tr.shape[0], n_float), columns=col_names)
    val_shadow = pd.DataFrame(np.random.randn(val.shape[0], n_float), columns=col_names)
    for i in range(n_cat_big):
        col_name = f'shadow_cat_big_{i+1}'
        tr_shadow[col_name] = pd.Series(np.random.choice(np.arange(200).astype(str), size=tr.shape[0], replace=True)).astype('category')
        val_shadow[col_name] = pd.Series(np.random.choice(np.arange(200).astype(str), size=val.shape[0], replace=True)).astype(tr_shadow[col_name].dtype)

    for i in range(n_cat_small):
        col_name = f'shadow_cat_small_{i+1}'
        tr_shadow[col_name] = pd.Series(np.random.choice(np.arange(4).astype(str), size=tr.shape[0], replace=True)).astype('category')
        val_shadow[col_name] = pd.Series(np.random.choice(np.arange(4).astype(str), size=val.shape[0], replace=True)).astype(tr_shadow[col_name].dtype)

    return tr_shadow, val_shadow


def train_linear_model(tr, features, target_col):
    X_tr = tr[features].select_dtypes(np.number).fillna(-1)
    logreg_features = X_tr.columns
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    model = LogisticRegression()
    model.fit(X_tr, tr[target_col])
    return model, logreg_features, scaler


def train_cb_model(tr, val, features, target_col, params=None, shadow_features=False):
    tr_shadow, val_shadow = pd.DataFrame(), pd.DataFrame()
    if shadow_features:
        tr_shadow, val_shadow = get_shadow_features(tr, val)

    X_tr = pd.concat([tr[features], tr_shadow], axis=1, sort=False)
    X_val = pd.concat([val[features], val_shadow], axis=1, sort=False)
    tr_cb = cb.Pool(X_tr, tr[target_col], cat_features=X_tr.select_dtypes('category').columns.to_list())
    val_cb = cb.Pool(X_val, val[target_col], cat_features=X_val.select_dtypes('category').columns.to_list())

    model = cb.CatBoost({
        'thread_count': 16,
        'loss_function': 'Logloss',
        'learning_rate': 0.01,
        'iterations': 500,
        'eval_metric': 'AUC',
        'verbose': 500,
        'early_stopping_rounds': 20,
    })
    model.fit(tr_cb, eval_set=val_cb)
    if shadow_features:
        return model, tr_shadow.columns, X_tr, X_val
    return model


def train_xgb_model(tr, val, features, target_col, params=None, shadow_features=False):

    tr_shadow, val_shadow = pd.DataFrame(), pd.DataFrame()
    if shadow_features:
        tr_shadow, val_shadow = get_shadow_features(tr, val)

    X_tr = pd.concat([tr[features], tr_shadow], axis=1, sort=False)
    X_val = pd.concat([val[features], val_shadow], axis=1, sort=False)
    tr_xgb = xgb.DMatrix(X_tr, tr[target_col], enable_categorical=True)
    val_xgb = xgb.DMatrix(X_val, val[target_col], enable_categorical=True)

    params_ = {
        'nthread': 16,
        'objective': 'binary:logistic',
        'learning_rate': 0.01,
        'metric': 'auc',
        'verbose': -1
    }
    if params is not None:
        params_.update(params)

    model = xgb.train(params_, tr_xgb, num_boost_round=300, evals=[(val_xgb, 'val_name')], early_stopping_rounds=100, verbose_eval=50)
    
    if shadow_features:
        return model, tr_shadow.columns, X_tr, X_val
    return model


def train_model(tr, val, features, target_col, params=None, shadow_features=False, sklearn_style=False):

    tr_shadow, val_shadow = pd.DataFrame(), pd.DataFrame()
    if shadow_features:
        tr_shadow, val_shadow = get_shadow_features(tr, val)

    X_tr = pd.concat([tr[features], tr_shadow], axis=1, sort=False)
    X_val = pd.concat([val[features], val_shadow], axis=1, sort=False)
    tr_lgb = lgb.Dataset(X_tr, tr[target_col])
    val_lgb = lgb.Dataset(X_val, val[target_col])

    params_ = {
        'nthread': 16,
        'objective': 'binary',
        'learning_rate': 0.01,
        'metric': 'auc',
        'verbose': -1
    }
    if params is not None:
        params_.update(params)

    if not sklearn_style:
        model = lgb.train(params_, tr_lgb, num_boost_round=300, valid_sets=[val_lgb], callbacks=[lgb.early_stopping(100)])
    else:
        model = lgb.LGBMClassifier(**params, n_estimators=300)
        model.fit(X_tr, tr[target_col], eval_set=[(X_val, val[target_col])], callbacks=[lgb.early_stopping(100)])
    
    if shadow_features:
        return model, tr_shadow.columns, X_tr, X_val
    return model


def get_different_scores(tr, val, features, target_col):
    model_lgb = train_model(tr, val, features, target_col)
    
    model_rf = train_model(
        tr, val, features, target_col, params={
        'boosting_type': 'rf',
        'bagging_fraction': 0.4,
        'colsample_bytree': 0.6,
        'bagging_freq': 1,
    })

    model_gbdt_pl = train_model(
        tr, val, features, target_col, params={
        'colsample_bytree': 0.9,
        'max_depth': 3,
        'linear_tree': True,
        'linear_lambda': 0.01
    })

    model_cb = train_cb_model(tr, val, features, target_col)

    model_logreg, logreg_features, scaler = train_linear_model(tr, features, target_col)

    res = pd.DataFrame({
        'plain_score': model_lgb.predict(val[features], raw_score=True),
        'rf_score': model_rf.predict(val[features], raw_score=True),
        'linear_score': model_gbdt_pl.predict(val[features], raw_score=True),
        'cb_score': model_cb.predict(val[features], prediction_type='RawFormulaVal'),
        'logreg_score': model_logreg.predict_proba(scaler.transform(val[logreg_features].fillna(-1)))[:, 1]
    })
    return res


def plot_scores_reg(model, X_val, y_val):
    y_pred_val_raw = model.predict(X_val, raw_score=True)
    sns.scatterplot(x=y_val, y=y_pred_val_raw, c=np.abs(y_val - y_pred_val_raw), cmap='coolwarm')
    sns.lineplot(x=y_val, y=y_val, color='red')
    plt.ylabel('model prediction')
    

def plot_scores(model, X_tr, y_tr, X_val, y_val, split_col=None, support_col=None, support_log=False):
    y_pred_tr_raw = model.predict(X_tr, raw_score=True)
    y_pred_val_raw = model.predict(X_val, raw_score=True)

    if split_col is not None:
        split_col_series_tr = X_tr[split_col]
        split_col_series_val = X_val[split_col]
        split_col_uniques = X_tr[split_col].unique()
    else:
        split_col_series_tr = pd.Series(np.ones(X_tr.shape[0]))
        split_col_series_val = pd.Series(np.ones(X_val.shape[0]))
        split_col_uniques = [1]
        
    for val in split_col_uniques:
        cond_tr = split_col_series_tr.eq(val) if not pd.isna(val) else split_col_series_tr.isnull()
        cond_val = split_col_series_val.eq(val) if not pd.isna(val) else split_col_series_val.isnull()

        if support_col is None:
            mosaic = [['tr', 'val']]
            fig, ax = plt.subplot_mosaic(mosaic, figsize=(8, 3))
            for key in ax:
                if split_col is not None:
                    fig.suptitle(f'{split_col}={val}')
                ax[key].set_title(key, fontsize=15)
    
            sns.histplot(x=y_pred_tr_raw[cond_tr], hue=y_tr[cond_tr], bins=33, ax=ax['tr'])
            sns.histplot(x=y_pred_val_raw[cond_val], hue=y_val[cond_val], bins=33, ax=ax['val'])
            fig.tight_layout()

        else:
            g = sns.JointGrid(X_val.assign(model_score=y_pred_val_raw).loc[cond_val], x='model_score', y=support_col, hue=y_val.loc[cond_val])
            g.plot_marginals(sns.histplot)
            g.plot_joint(sns.scatterplot, s=6)
            g.plot_joint(sns.kdeplot, gridsize=30, bw_adjust=0.5)
            if support_log:
                ax = g.ax_joint
                ax.set_yscale('log')

        
    plt.show()


def get_split(df, val_size=0.33):
    train_idx = np.random.choice(df.index, size=int(df.shape[0]*(1-val_size)), replace=False)
    val_idx = np.setdiff1d(df.index, train_idx)
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)


def get_df_info(df, cols=None):
    if cols is None:
        cols = df.columns

    def mode_extractor(df, col):
        vc = df.loc[~df[col].isin((0, "")) & ~df[col].isnull()][col].value_counts(dropna=False, normalize=True)
        mode = vc.idxmax()
        frac = vc.loc[mode]
        return mode, round(frac, 2)

    mode_info = [mode_extractor(df, col) for col in cols]

    def example_extractor(df, col):
        unique_els = df[col].unique()
        unique_els = unique_els[~pd.isna(unique_els)]
        ex_1, ex_2 = np.random.choice(unique_els, size=2, replace=False)
        return ex_1, ex_2
        
    examples = [example_extractor(df, col) for col in cols]
    
    res = pd.DataFrame({
        'data_type': df[cols].dtypes,
        'n_unique': df[cols].nunique(dropna=False),
        'example_1': map(lambda x: x[0], examples),
        'example_2': map(lambda x: x[1], examples),
        'nan_frac': df[cols].isnull().mean(axis=0),
        'zero_frac': df[cols].eq(0).mean(axis=0),
        'empty_frac': df[cols].eq('').mean(axis=0),
        'mode_el': map(lambda x: x[0], mode_info),
        'mode_frac': map(lambda x: x[1], mode_info),
    })
    tmp_max = res[['nan_frac', 'zero_frac', 'empty_frac', 'mode_frac']].max(axis=1)
    res['trash_score'] = np.maximum(res.mode_frac - tmp_max, tmp_max)
    res.nan_frac = res.nan_frac.replace(0, '∅')
    res.zero_frac = res.zero_frac.replace(0, '∅')
    res.empty_frac = res.empty_frac.replace(0, '∅')
    
    return (
            res.sort_values('trash_score', ascending=False)
            .style
            .format(precision=2)
            .set_properties(subset=['trash_score'], **{'font-weight': 'bold'})
    )


def get_df_info_new(df: pd.DataFrame, /, thr=0.5, *args, **kwargs) -> pd.DataFrame:
    '''
    Описание:
    Функция выводит статистику о колонках датафрейма в виде датафрейма.
    Результат содержит информацию о типе данных, об уникальных значениях,
    о количестве NaN, нулей и пустых строк,
    о максимальном значении и его частоте,
    два различных примера данных,
    степень адекватности фичи.
    ...

    df: исходный датафрейм,
    thr: граничное значения для опредления адекватность фичи.
    ...

    returns: pd.DataFrame со сборной информацией о датафрейме.
    '''
    
    df_size = df.shape[0]
    columns = df.columns
    numeric_columns = df.select_dtypes(np.number).columns
    string_columns = np.concatenate((df.select_dtypes('string').columns, df.select_dtypes('object').columns))
    result_columns = ["datatype", "nunique", "nan", "zero", "empty_str", "vc_max", "vc_max_freq", "example_1", "example_2", "trash_score"]
    result_dict = dict(zip(result_columns, [[] for _ in range(len(result_columns))]))
    
    for column in columns:
        
        result_dict["datatype"].append(df[column].dtype.name)
        result_dict["nunique"].append(df[column].unique().size)
        
        tmp = df[column].isna().sum()
        push_nan_tmp = tmp / df_size
        push = f"n: {push_nan_tmp: 0.3f}" if tmp > 0 else "-1"
        result_dict["nan"].append(push)

        tmp = column in numeric_columns
        if tmp:
            tmp = df[column].eq(0).sum()
            push_zero_tmp = tmp / df_size
            push = f"z: {push_zero_tmp: 0.3f}" if tmp > 0 else "-1"
        else:
            push_zero_tmp = 0
            push = "-1"
        result_dict["zero"].append(push)

        tmp = column in string_columns
        if tmp:
            tmp = df[column].eq("").sum()
            push_empty_str_tmp = tmp / df_size
            push = f"e: {push_empty_str_tmp: 0.3f}" if tmp > 0 else "-1"
        else:
            push_empty_str_tmp = 0
            push = "-1"
        result_dict["empty_str"].append(push)

        push = df[column].value_counts(dropna=True)
        push_index = push.index[0]
        push_freq = push.values[0]
        result_dict["vc_max"].append(push_index)
        push_vc_max_freq_tmp = push_freq / df_size
        result_dict["vc_max_freq"].append(f"n: {push_vc_max_freq_tmp: 0.3f}")

        tmp = df[column].dropna().unique()
        match tmp.size:
            case 0:
                push = 2 * ["No example"]
            case 1:
                push = [tmp[0], "No example"]
            case _:
                push = [tmp[0], tmp[1]]
        result_dict["example_1"].append(push[0])
        result_dict["example_2"].append(push[1])

        tmp = max(
            push_nan_tmp + push_zero_tmp + push_empty_str_tmp,\
            push_vc_max_freq_tmp if push_vc_max_freq_tmp > thr else 0,
        )
        push = f"{tmp: 0.3f}" if tmp > 0 else "-1"
        result_dict["trash_score"].append(push)

    result_df = pd.DataFrame(result_dict, index=columns).sort_values("trash_score", key=lambda x: -np.float_(x))
    
    return result_df

N_UNIQUE_THRESHOLD = 55

def get_categorical_features(df, nunique_threshold=N_UNIQUE_THRESHOLD):
    return [feature for feature in df.columns 
            if df[feature].nunique() < nunique_threshold]

def get_numerical_features(df, nunique_threshold=N_UNIQUE_THRESHOLD):
    return [feature for feature in df.select_dtypes(include=[np.number]).columns 
            if df[feature].nunique() >= nunique_threshold]

def build_my_info_table(df, nunique_threshold=N_UNIQUE_THRESHOLD):
    # Check for an empty DataFrame
    if df is None or df.empty:
        return None
    # Convert boolean columns to integer inplace
    boolean_columns = df.select_dtypes(include='bool').columns
    df[boolean_columns] = df[boolean_columns].astype(int)
    # Select numerical columns
    numerical_features = get_numerical_features(df)
    # Initialize list to store feature-wise metrics
    metrics = []
    for idx, col in enumerate(df.columns):
        column_data = df[col]
        dtype   = column_data.dtypes
        count   = column_data.count()
        mean    = column_data.mean()   if col in numerical_features else ''
        std     = column_data.std()    if col in numerical_features else ''
        min_val = column_data.min()    if col in numerical_features else ''
        q25     = column_data.quantile(0.25) if col in numerical_features else ''
        median  = column_data.median() if col in numerical_features else ''
        q75     = column_data.quantile(0.75) if col in numerical_features else ''
        q95     = column_data.quantile(0.95) if col in numerical_features else ''
        max_val = column_data.max()    if col in numerical_features else ''
        iqr     = max_val - min_val    if col in numerical_features else ''
        nunique = column_data.nunique()
        unique_values   = column_data.unique() if nunique < nunique_threshold else ''
        mode    = column_data.mode().iloc[0] if not column_data.mode().empty else ''
        mode_count      = column_data.value_counts().max() \
                                             if not column_data.value_counts().empty else ''
        mode_percentage = (round(mode_count * 100 / len(column_data), 1) 
                                             if mode_count not in ['', None] else '')
        null_count      = column_data.isnull().sum()
        null_percentage = round(column_data.isnull().mean() * 100, 1)
        # Append the calculated metrics to the list
        metrics.append({
            "#": idx,
            "column": col,
            "dtype": dtype,
            "count": count,
            "mean": round(mean, 1)   if mean    not in ['', None] else '',
            "std": round(std, 1)     if std     not in ['', None] else '',
            "min": round(min_val, 1) if min_val not in ['', None] else '',
            "25%": round(q25, 1)     if q25     not in ['', None] else '',
            "50%": round(median, 1)  if median  not in ['', None] else '',
            "75%": round(q75, 1)     if q75     not in ['', None] else '',
            "95%": round(q95, 1)     if q95     not in ['', None] else '',
            "max": round(max_val, 1) if max_val not in ['', None] else '',
            "IQR": round(iqr, 1)     if iqr     not in ['', None] else '',
            "nunique": nunique,
            "unique": unique_values,
            "mode": mode,
            "mode #": mode_count,
            "mode %": mode_percentage,
            "null #": null_count,
            "null %": null_percentage,
        })
    # Convert metrics list to DataFrame
    df_info = pd.DataFrame(metrics)
    # Ensure sorting by dtype is stable
    df_info = df_info.sort_values(by='dtype').reset_index(drop=True)
    return df_info

def filter_users_by_99_percentile(df, user_id_col='user_id', target_col='target'):
    # Колонки, которые НЕ будем учитывать при фильтрации:
    # 1. user_id и target
    # 2. Колонки, начинающиеся с "days"
    exclude_cols = [user_id_col, target_col] + [col for col in df.columns if col.startswith('days')]
    
    # Вычисляем 99-й персентиль только для оставшихся колонок
    percentile_99 = df.drop(columns=exclude_cols).quantile(0.99)
    
    # Маска: True, если ВСЕ значения в строке <= 99% персентиля
    mask = (df.drop(columns=exclude_cols) <= percentile_99).all(axis=1)
    
    # Возвращаем только подходящие user_id
    filtered_users = df.loc[mask, user_id_col]
    
    # Фильтруем исходный датафрейм
    filtered_df = df[df[user_id_col].isin(filtered_users)]
    
    return filtered_df


# Функция для создания полиномиальных фичей с исправлением
def add_polynomial_features_pl(df, features, degree=2):
    """
    Добавляет полиномиальные комбинации фичей до указанной степени
    """
    # Создаем копию датафрейма для модификации
    result_df = df.clone()
    
    # Создаем все комбинации фичей для полиномов
    for feat1, feat2 in combinations(features, 2):
        # Умножение фичей (взаимодействие)
        result_df = result_df.with_columns(
            (pl.col(feat1) * pl.col(feat2)).alias(f'{feat1}_x_{feat2}')
        )
        
        # Можно добавить другие степени при необходимости
        if degree > 2:
            result_df = result_df.with_columns(
                (pl.col(feat1)**2 * pl.col(feat2)).alias(f'{feat1}^2_x_{feat2}'),
                (pl.col(feat1) * pl.col(feat2)**2).alias(f'{feat1}_x_{feat2}^2')
            )
    
    # Добавляем квадраты фичей
    for feat in features:
        result_df = result_df.with_columns(
            (pl.col(feat)**2).alias(f'{feat}^2')
        )
        if degree > 2:
            result_df = result_df.with_columns((pl.col(feat)**3).alias(f'{feat}^3'))
    
    return result_df


def add_polynomial_features_pd(df, features, degree=2):
    """
    Добавляет полиномиальные комбинации фичей до указанной степени
    для pandas DataFrame
    """
    # Создаем копию датафрейма для модификации
    result_df = df.copy()
    
    # Создаем все комбинации фичей для полиномов
    for feat1, feat2 in combinations(features, 2):
        # Умножение фичей (взаимодействие)
        result_df[f'{feat1}_x_{feat2}'] = result_df[feat1] * result_df[feat2]
        
        # Добавляем степени выше 2-й
        if degree > 2:
            result_df[f'{feat1}^2_x_{feat2}'] = result_df[feat1]**2 * result_df[feat2]
            result_df[f'{feat1}_x_{feat2}^2'] = result_df[feat1] * result_df[feat2]**2
    
    # Добавляем квадраты и кубы фичей
    for feat in features:
        result_df[f'{feat}^2'] = result_df[feat]**2
        if degree > 2:
            result_df[f'{feat}^3'] = result_df[feat]**3
    
    return result_df

import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE


def cluster_search_queries(
    data: pd.DataFrame,
    text_column: str = "search_query",
    sample_size: int = 10_000,
    algorithm: str = "gmm",
    dim_reduction: str = "pca",
    n_clusters: int = 10,
    random_state: int = 42,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Кластеризует текстовые запросы и визуализирует результат.
    
    Параметры:
    ----------
    data : pd.DataFrame
        DataFrame с текстовыми данными.
    text_column : str, optional
        Название колонки с текстом (по умолчанию "search_query").
    sample_size : int, optional
        Сколько строк брать (если None — берутся все).
    algorithm : str, optional
        Алгоритм кластеризации ("gmm" или "spectral").
    dim_reduction : str, optional
        Метод уменьшения размерности ("pca" или "tsne").
    n_clusters : int, optional
        Количество кластеров.
    random_state : int, optional
        Seed для воспроизводимости.
    plot : bool, optional
        Строить ли график (по умолчанию True).
    
    Возвращает:
    ----------
    pd.DataFrame
        Исходный DataFrame с добавленной колонкой 'cluster'.
    """
    
    # Подготовка данных (удаление NaN и выборка)
    df = data.dropna(subset=[text_column])
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    
    # Векторизация текста (TF-IDF)
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df[text_column])
    
    # Уменьшение размерности
    if dim_reduction == "pca":
        X_reduced = PCA(n_components=2, random_state=random_state).fit_transform(X.toarray())
    elif dim_reduction == "tsne":
        X_reduced = TSNE(n_components=2, perplexity=30, random_state=random_state).fit_transform(X.toarray())
    else:
        raise ValueError("Допустимые методы уменьшения размерности: 'pca' или 'tsne'")
    
    # Кластеризация
    if algorithm == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=random_state
        )
        df['cluster'] = model.fit_predict(X_reduced)
    elif algorithm == "spectral":
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=10,
            random_state=random_state
        )
        df['cluster'] = model.fit_predict(X_reduced)
    else:
        raise ValueError("Допустимые алгоритмы: 'gmm' или 'spectral'")
    
    # Визуализация (если нужно)
    if plot:
        fig = px.scatter(
            df,
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            color='cluster',
            hover_data=[text_column],
            title=f'Кластеризация ({algorithm.upper()} + {dim_reduction.upper()})',
            width=1200,
            height=800
        )
        fig.update_traces(
            marker=dict(size=4, opacity=0.7, line=dict(width=0.2)),
            selector=dict(mode='markers')
        )
        fig.update_layout(
            hoverlabel=dict(bgcolor="white", font_size=12),
            plot_bgcolor='rgba(240,240,240,0.9)'
        )
        fig.show()


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def add_knn_features(df_pd, features, target, n_neighbors=5):
    df = df_pd.copy()
    
    # Определяем стратегии заполнения пропусков
    zero_fill_cols = [col for col in features if col.startswith(('num_', 'sum_', 'max_', 'days_'))]
    time_fill_cols = [col for col in features if col.endswith('_time')]
    
    # Заполняем пропуски
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)
    df[time_fill_cols] = df[time_fill_cols].fillna(pd.Timestamp('2024-01-01 00:00:00'))

    print('nans in knn filled')
    
    # Масштабируем данные
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    print('data in knn scaled')
    
    # KNN для поиска ближайших соседей
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    print('knn created')
    knn.fit(df[features])
    print('knn fitted')
    
    distances, indices = knn.kneighbors(df[features])
    print('knn build ended')

    # Создаем новые признаки
    knn_features = {}
    
    # Метрики расстояний
    df['knn_distance_mean'] = distances.mean(axis=1)
    df['knn_distance_max'] = distances.max(axis=1)
    df['knn_distance_min'] = distances.min(axis=1)
    df['knn_distance_std'] = distances.std(axis=1)
    df['knn_distance_range'] = df['knn_distance_max'] - df['knn_distance_min']

    for feature in features:
        knn_values = df[feature].values[indices]

        knn_features[f'knn_{feature}_mean'] = knn_values.mean(axis=1)
        knn_features[f'knn_{feature}_max'] = knn_values.max(axis=1)
        knn_features[f'knn_{feature}_min'] = knn_values.min(axis=1)
        knn_features[f'knn_{feature}_std'] = knn_values.std(axis=1)
        knn_features[f'knn_{feature}_median'] = np.median(knn_values, axis=1)
        knn_features[f'knn_{feature}_sum'] = knn_values.sum(axis=1)
        knn_features[f'knn_{feature}_range'] = knn_features[f'knn_{feature}_max'] - knn_features[f'knn_{feature}_min']
    
    # Среднее и стандартное отклонение таргета среди соседей
    target_neighbors = df[target].values[indices]
    df['knn_target_mean'] = target_neighbors.mean(axis=1)
    df['knn_target_std'] = target_neighbors.std(axis=1)
    df['knn_target_median'] = np.median(target_neighbors, axis=1)
    df['knn_target_sum'] = target_neighbors.sum(axis=1)
    df['knn_target_range'] = df['knn_target_mean'] - df['knn_target_min']

    # Добавляем новые признаки в DataFrame
    for key, value in knn_features.items():
        df[key] = value

    print('knn features created, end of knn')
    
    return df


import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler

def add_knn_features_faiss(df_pd, features, target, n_neighbors=5, use_gpu=True):
    df = df_pd.copy()
    
    # Заполнение пропусков
    zero_fill_cols = [col for col in features if col.startswith(('num_', 'sum_', 'max_', 'days_'))]
    time_fill_cols = [col for col in features if col.endswith('_time')]
    
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)
    df[time_fill_cols] = df[time_fill_cols].fillna(pd.Timestamp('2024-01-01 00:00:00'))
    print('Nans filled')
    
    # Масштабирование
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    print('Data scaled')
    
    # FAISS индекс
    d = len(features)
    data = df[features].values.astype('float32')
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
        print('Using GPU')
    else:
        index = faiss.IndexFlatL2(d)
        print('Using CPU')
    
    index.add(data)
    print('FAISS index built')
    
    # Поиск ближайших соседей
    distances, indices = index.search(data, n_neighbors)
    print('KNN search done')
    
    # Создание новых признаков
    knn_features = {}
    df['knn_distance_mean'] = distances.mean(axis=1)
    df['knn_distance_max'] = distances.max(axis=1)
    df['knn_distance_min'] = distances.min(axis=1)
    df['knn_distance_std'] = distances.std(axis=1)
    df['knn_distance_range'] = df['knn_distance_max'] - df['knn_distance_min']
    
    for feature in features:
        knn_values = df[feature].values[indices]
        knn_features[f'knn_{feature}_mean'] = knn_values.mean(axis=1)
        knn_features[f'knn_{feature}_max'] = knn_values.max(axis=1)
        knn_features[f'knn_{feature}_min'] = knn_values.min(axis=1)
        knn_features[f'knn_{feature}_std'] = knn_values.std(axis=1)
        knn_features[f'knn_{feature}_median'] = np.median(knn_values, axis=1)
        knn_features[f'knn_{feature}_sum'] = knn_values.sum(axis=1)
        knn_features[f'knn_{feature}_range'] = knn_features[f'knn_{feature}_max'] - knn_features[f'knn_{feature}_min']
    
    if target is not None:
        target_neighbors = df[target].values[indices]
        df['knn_target_mean'] = target_neighbors.mean(axis=1)
        df['knn_target_std'] = target_neighbors.std(axis=1)
        df['knn_target_median'] = np.median(target_neighbors, axis=1)
        df['knn_target_sum'] = target_neighbors.sum(axis=1)
        df['knn_target_range'] = df['knn_target_mean'] - target_neighbors.min(axis=1)
    
    # Добавляем новые признаки в DataFrame
    for key, value in knn_features.items():
        df[key] = value
    
    print('KNN features created')
    return df


import polars as pl
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.decomposition import TruncatedSVD

import polars as pl
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.decomposition import TruncatedSVD
from itertools import combinations

def old_generate_features(actions_history, search_history, product_info, widget_info, end_date, prediction_date):

    """
    Генерация расширенных признаков с учётом полной структуры данных
    
    Параметры:
        actions_history: DataFrame с историей действий
        search_history: DataFrame с историей поиска
        product_info: DataFrame с информацией о товарах
        widget_info: DataFrame с информацией о виджетах
        end_date: Дата окончания периода для генерации признаков
        prediction_date: Дата, относительно которой считаются временные признаки
    """
    # 1. Convert all inputs to Polars DataFrames if they aren't already
    actions_history = pl.DataFrame(actions_history)
    search_history = pl.DataFrame(search_history)
    product_info = pl.DataFrame(product_info)
    widget_info = pl.DataFrame(widget_info)
    
    # 2. Verify column names exist before processing
    required_columns = {
        'actions_history': ['user_id', 'action_type_id', 'product_id', 'timestamp', 'page_product_id', 'widget_name_id'],
        'search_history': ['user_id', 'search_query', 'timestamp', 'widget_name_id'],
        'product_info': ['product_id', 'discount_price', 'price', 'category_id', 'brand'],
        'widget_info': ['widget_name_id', 'widget_name']
    }
    
    for df_name, cols in required_columns.items():
        df = locals()[df_name]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {df_name}: {missing}")

    # Rest of your function remains the same...
    widget_names = widget_info.to_pandas().set_index('widget_name_id')['widget_name'].to_dict()
    product_info = product_info.with_columns(
        discount_percent=(1 - pl.col('discount_price')/pl.col('price')).fill_nan(0)
    )
    
    # Основные агрегации
    actions = actions_history.filter(pl.col('timestamp').dt.date() <= end_date)
    searches = search_history.filter(pl.col('timestamp').dt.date() <= end_date)
    
    # 1. Базовые агрегации по действиям
    action_features = []
    action_types = {
        1: "click",
        2: "favorite",
        3: "order",
        5: "to_cart"
    }
    
    for action_id, action_name in action_types.items():
        action_data = actions.filter(pl.col('action_type_id') == action_id)
        
        # Агрегации по товарам
        product_aggs = (
            action_data
            .join(product_info, on='product_id')
            .group_by('user_id')
            .agg(
                # Количественные метрики
                pl.count().alias(f'{action_name}_count'),
                pl.n_unique('product_id').alias(f'{action_name}_unique_products'),
                pl.sum('discount_price').alias(f'{action_name}_total_value'),
                pl.mean('discount_price').alias(f'{action_name}_avg_value'),
                
                # Временные метрики
                (pl.lit(prediction_date) - pl.col('timestamp').dt.date()).dt.total_days().alias(f'days_since_last_{action_name}'),
                (pl.lit(prediction_date) - pl.col('timestamp').dt.date()).dt.total_days().alias(f'days_since_first_{action_name}'),
                
                # Метрики по скидкам
                pl.mean('discount_percent').alias(f'{action_name}_avg_discount'),
                pl.max('discount_percent').alias(f'{action_name}_max_discount'),
                
                # Особые метрики для заказов
                *([
                    pl.median('price').alias(f'{action_name}_median_price'),
                    (pl.col('discount_price')/pl.col('price')).mean().alias(f'{action_name}_discount_ratio'),
                ] if action_id == 3 else [])
            )
        )

        
        # Агрегации по виджетам
        all_users = actions.select('user_id').unique()
        
        # Get counts per widget per user
        widget_counts = (
            action_data
            .group_by(['user_id', 'widget_name_id'])
            .agg(pl.count().alias('count'))
        )
        
        # Get all widget IDs from widget_info
        all_widget_ids = widget_info['widget_name_id'].unique().to_list()
        
        # Create a DataFrame with all possible combinations
        # This ensures we have all widgets for all users
        full_grid = (
            all_users
            .join(
                pl.DataFrame({'widget_name_id': all_widget_ids}),
                how='cross'
            )
        )
        
        # Left join with our counts
        widget_aggs = (
            full_grid
            .join(
                widget_counts,
                on=['user_id', 'widget_name_id'],
                how='left'
            )
            .fill_null(0)
            .pivot(
                index='user_id',
                columns='widget_name_id',
                values='count',
                aggregate_function='first'  # Since we already aggregated
            )
            .rename({
                str(wid): f'{action_name}_widget_{widget_names.get(wid, wid)}'
                for wid in all_widget_ids
            })
        )
        
        action_features.extend([product_aggs, widget_aggs])
    
    # 2. Особые признаки для заказов (покупок)
    order_features = (
        actions
        .filter(pl.col('action_type_id') == 3)
        .join(product_info, on='product_id')
        .group_by('user_id')
        .agg(
            # Повторные покупки
            (pl.count() - pl.n_unique('product_id')).alias('repeat_orders'),
            
            # Разнообразие категорий
            pl.n_unique('category_id').alias('unique_categories'),
            pl.n_unique('brand').alias('unique_brands'),
            
            # Статистики по времени между заказами
            pl.col('timestamp').diff().dt.hours().mean().alias('avg_hours_between_orders'),
            
            # RFM метрики
            (pl.lit(prediction_date) - pl.col('timestamp').dt.date()).dt.total_days().alias('recency'),
            pl.count().alias('frequency'),
            pl.sum('discount_price').alias('monetary'),
        )
        .with_columns(
            (pl.col('recency') * 0.4 + pl.col('frequency') * 0.3 + pl.col('monetary') * 0.3).alias('rfm_score')
        )
    )
    action_features.append(order_features)
    
    # 3. Признаки из поиска
    search_features = [
        searches
        .group_by('user_id')
        .agg(
            pl.count().alias('total_searches'),
            pl.n_unique('search_query').alias('unique_searches'),
            (prediction_date - pl.col('timestamp').max().dt.date()).dt.days().alias('days_since_last_search'),
        ),
        
        searches
        .group_by(['user_id', 'widget_name_id'])
        .agg(pl.count().alias('count'))
        .pivot(index='user_id', columns='widget_name_id', values='count', aggregate_function='sum')
        .rename({str(k): f'search_widget_{v}' for k, v in widget_names.items()})
        .fill_null(0)
    ]
    
    # 4. Взаимодействия товаров (через page_product_id)
    interaction_features = (
        actions
        .filter(pl.col('page_product_id').is_not_null())
        .join(product_info.select(['product_id', 'category_id']), on='product_id')
        .join(
            product_info.select(['product_id', 'category_id']).rename({'product_id': 'page_product_id', 'category_id': 'page_category_id'}),
            on='page_product_id'
        )
        .group_by('user_id')
        .agg(
            # Сколько раз смотрели товары из той же категории
            (pl.col('category_id') == pl.col('page_category_id')).sum().alias('same_category_interactions'),
            
            # Среднее количество взаимодействий на товар
            pl.count() / pl.n_unique('page_product_id').alias('avg_interactions_per_product'),
        )
    )
    action_features.append(interaction_features)
    
    # 5. Временные паттерны
    time_features = (
        actions
        .group_by('user_id')
        .agg(
            pl.col('timestamp').dt.hour().mode().first().alias('most_active_hour'),
            pl.col('timestamp').dt.weekday().mode().first().alias('most_active_weekday'),
            
            # Активность в разные периоды
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=7))).sum().alias('actions_last_7d'),
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=30))).sum().alias('actions_last_30d'),
            
            # Соотношение активности
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=7))).sum() / 
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=30))).sum().alias('recent_activity_ratio')
        )
    )
    action_features.append(time_features)
    
    # 6. SVD разложение для товаров (user-item матрица)
    user_item_matrix = (
        actions
        .filter(pl.col('action_type_id') == 3)
        .group_by(['user_id', 'product_id'])
        .agg(pl.count().alias('count'))
        .pivot(index='user_id', columns='product_id', values='count', aggregate_function='sum')
        .fill_null(0)
    )
    
    svd = TruncatedSVD(n_components=5, random_state=42)
    svd_features = svd.fit_transform(user_item_matrix.drop('user_id').to_pandas())
    
    svd_df = pl.DataFrame({
        'user_id': user_item_matrix['user_id'],
        **{f'svd_{i+1}': svd_features[:, i] for i in range(5)}
    })
    action_features.append(svd_df)
    
    # Собираем все признаки вместе
    features = action_features[0]
    for df in action_features[1:] + search_features:
        features = features.join(df, on='user_id', how='left')
    
    # Заполняем пропуски
    features = features.fill_null(0)
    
    # 7. Создаем взаимодействия признаков
    numeric_cols = [col for col in features.columns if col not in ['user_id'] and features[col].dtype in [pl.Float64, pl.Int64]]
    
    for col1, col2 in combinations(numeric_cols, 2):
        if f'{col1}_x_{col2}' not in features.columns:
            features = features.with_columns([
                (pl.col(col1) * pl.col(col2)).alias(f'{col1}_x_{col2}'),
                (pl.col(col1) + pl.col(col2)).alias(f'{col1}_plus_{col2}'),
                (pl.col(col1) / (pl.col(col2) + 1e-6)).alias(f'{col1}_div_{col2}'),
            ])
    
    return features


import polars as pl
from datetime import timedelta
from itertools import combinations
from sklearn.decomposition import TruncatedSVD

def validate_input_dataframes(actions_history, search_history, product_info, widget_info):
    """
    Проверка входных данных на наличие необходимых колонок
    """
    required_columns = {
        'actions_history': ['user_id', 'action_type_id', 'product_id', 'timestamp', 'page_product_id', 'widget_name_id'],
        'search_history': ['user_id', 'search_query', 'timestamp', 'widget_name_id'],
        'product_info': ['product_id', 'discount_price', 'price', 'category_id', 'brand'],
        'widget_info': ['widget_name_id', 'widget_name']
    }
    
    for df_name, cols in required_columns.items():
        df = locals()[df_name]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {df_name}: {missing}")

def prepare_basic_data(actions_history, search_history, product_info, widget_info, end_date):
    """
    Подготовка базовых данных и словаря виджетов
    """
    widget_names = widget_info.to_pandas().set_index('widget_name_id')['widget_name'].to_dict()
    product_info = product_info.with_columns(
        discount_percent=(1 - pl.col('discount_price')/pl.col('price')).fill_nan(0)
    )
    
    actions = actions_history.filter(pl.col('timestamp').dt.date() <= end_date)
    searches = search_history.filter(pl.col('timestamp').dt.date() <= end_date)
    
    return actions, searches, product_info, widget_names

def generate_action_features(actions, product_info, widget_info, widget_names, prediction_date):
    """
    1. Генерация признаков на основе действий пользователей
    """
    action_features = []
    action_types = {
        1: "click",
        2: "favorite",
        3: "order",
        5: "to_cart"
    }
    
    for action_id, action_name in action_types.items():
        action_data = actions.filter(pl.col('action_type_id') == action_id)
        
        # Агрегации по товарам
        product_aggs = (
            action_data
            .join(product_info, on='product_id')
            .group_by('user_id')
            .agg(
                pl.count().alias(f'{action_name}_count'),
                pl.n_unique('product_id').alias(f'{action_name}_unique_products'),
                pl.sum('discount_price').alias(f'{action_name}_total_value'),
                pl.mean('discount_price').alias(f'{action_name}_avg_value'),
                (pl.lit(prediction_date) - pl.col('timestamp').dt.date()).dt.total_days().alias(f'days_since_last_{action_name}'),
                (pl.lit(prediction_date) - pl.col('timestamp').dt.date()).dt.total_days().alias(f'days_since_first_{action_name}'),
                pl.mean('discount_percent').alias(f'{action_name}_avg_discount'),
                pl.max('discount_percent').alias(f'{action_name}_max_discount'),
                *([
                    pl.median('price').alias(f'{action_name}_median_price'),
                    (pl.col('discount_price')/pl.col('price')).mean().alias(f'{action_name}_discount_ratio'),
                ] if action_id == 3 else [])
            )
        )

        # Агрегации по виджетам
        all_users = actions.select('user_id').unique()
        widget_counts = action_data.group_by(['user_id', 'widget_name_id']).agg(pl.count().alias('count'))
        all_widget_ids = widget_info['widget_name_id'].unique().to_list()
        
        full_grid = all_users.join(pl.DataFrame({'widget_name_id': all_widget_ids}), how='cross')
        
        widget_aggs = (
            full_grid
            .join(widget_counts, on=['user_id', 'widget_name_id'], how='left')
            .fill_null(0)
            .pivot(index='user_id', columns='widget_name_id', values='count', aggregate_function='first')
            .rename({
                str(wid): f'{action_name}_widget_{widget_names.get(wid, wid)}'
                for wid in all_widget_ids
            })
        )
        
        action_features.extend([product_aggs, widget_aggs])
    
    return action_features

def generate_order_features(actions, product_info, prediction_date):
    """
    2. Генерация специальных признаков для заказов (покупок)
    """
    # Сначала получаем базовые агрегации
    order_stats = (
        actions
        .filter(pl.col('action_type_id') == 3)
        .join(product_info, on='product_id')
        .group_by('user_id')
        .agg(
            (pl.count() - pl.n_unique('product_id')).alias('repeat_orders'),
            pl.n_unique('category_id').alias('unique_categories'),
            pl.n_unique('brand').alias('unique_brands'),
            pl.col('timestamp').diff().dt.total_hours().mean().alias('avg_hours_between_orders'),
            pl.count().alias('frequency'),
            pl.sum('discount_price').alias('monetary')
        )
    )
    
    # Отдельно вычисляем recency (минимальное количество дней с последнего заказа)
    recency_stats = (
        actions
        .filter(pl.col('action_type_id') == 3)
        .group_by('user_id')
        .agg(
            (pl.lit(prediction_date) - pl.col('timestamp').dt.date()).min().dt.total_days().alias('recency')
        )
    )
    
    # Объединяем и вычисляем RFM
    return (
        order_stats
        .join(recency_stats, on='user_id')
        .with_columns(
            (pl.col('recency') * 0.4 + pl.col('frequency') * 0.3 + pl.col('monetary') * 0.3).alias('rfm_score')
        )
    )

def generate_search_features(searches, widget_names, prediction_date):
    """
    3. Generate search history features with proper error handling
    """
    # 1. Validate input columns
    required_columns = {'user_id', 'timestamp', 'widget_name_id'}
    missing_columns = required_columns - set(searches.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in search history: {missing_columns}")

    # 2. Convert prediction_date if needed
    if not isinstance(prediction_date, (pl.Expr, pl.Date, pl.Datetime)):
        prediction_date = pl.lit(prediction_date)

    # 3. Basic search statistics
    basic_stats = (
        searches
        .group_by('user_id')
        .agg(
            pl.count().alias('total_searches'),
            pl.n_unique('search_query').alias('unique_searches'),
            (prediction_date - pl.col('timestamp').max().dt.date()).cast(pl.Int64).alias('days_since_last_search')
        )
    )

    # 4. Widget-specific features (with safe pivot)
    try:
        # Get unique widget IDs present in both data and mapping
        valid_widgets = set(searches['widget_name_id'].unique()) & set(widget_names.keys())
        
        widget_stats = (
            searches
            .filter(pl.col('widget_name_id').is_in(list(valid_widgets)))
            .group_by(['user_id', 'widget_name_id'])
            .agg(pl.count().alias('count'))
            .pivot(
                index='user_id',
                columns='widget_name_id',
                values='count',
                aggregate_function='sum'
            )
            .rename({str(k): f'search_widget_{v}' for k, v in widget_names.items() 
                    if k in valid_widgets})
            .fill_null(0)
        )
    except Exception as e:
        print(f"Warning: Could not generate widget stats - {str(e)}")
        widget_stats = searches.select('user_id').unique()

    return basic_stats.join(widget_stats, on='user_id', how='left')

def generate_interaction_features(actions, product_info):
    """
    4. Генерация признаков взаимодействия товаров
    """
    return (
        actions
        .filter(pl.col('page_product_id').is_not_null())
        .join(product_info.select(['product_id', 'category_id']), on='product_id')
        .join(
            product_info.select(['product_id', 'category_id']).rename({'product_id': 'page_product_id', 'category_id': 'page_category_id'}),
            on='page_product_id'
        )
        .group_by('user_id')
        .agg(
            (pl.col('category_id') == pl.col('page_category_id')).sum().alias('same_category_interactions'),
            pl.count() / pl.n_unique('page_product_id').alias('avg_interactions_per_product'),
        )
    )

def generate_time_features(actions, end_date):
    """
    5. Генерация временных признаков активности
    """
    return (
        actions
        .group_by('user_id')
        .agg(
            pl.col('timestamp').dt.hour().mode().first().alias('most_active_hour'),
            pl.col('timestamp').dt.weekday().mode().first().alias('most_active_weekday'),
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=7))).sum().alias('actions_last_7d'),
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=30))).sum().alias('actions_last_30d'),
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=7))).sum() / 
            (pl.col('timestamp').dt.date() >= (end_date - timedelta(days=30))).sum().alias('recent_activity_ratio')
        )
    )

def generate_svd_features(actions):
    """
    6. Генерация SVD признаков из user-item матрицы
    """
    user_item_matrix = (
        actions
        .filter(pl.col('action_type_id') == 3)
        .group_by(['user_id', 'product_id'])
        .agg(pl.count().alias('count'))
        .pivot(index='user_id', columns='product_id', values='count', aggregate_function='sum')
        .fill_null(0)
    )
    
    svd = TruncatedSVD(n_components=5, random_state=42)
    svd_features = svd.fit_transform(user_item_matrix.drop('user_id').to_pandas())
    
    return pl.DataFrame({
        'user_id': user_item_matrix['user_id'],
        **{f'svd_{i+1}': svd_features[:, i] for i in range(5)}
    })

def generate_features(actions_history, search_history, product_info, widget_info, end_date, prediction_date):
    """
    Основная функция генерации признаков, объединяющая все подфункции
    """
    # Конвертация и валидация данных
    actions_history = pl.DataFrame(actions_history)
    search_history = pl.DataFrame(search_history)
    product_info = pl.DataFrame(product_info)
    widget_info = pl.DataFrame(widget_info)
    
    validate_input_dataframes(actions_history, search_history, product_info, widget_info)
    
    # Подготовка данных
    actions, searches, product_info, widget_names = prepare_basic_data(
        actions_history, search_history, product_info, widget_info, end_date
    )
    
    # Генерация всех признаков
    #action_features = generate_action_features(actions, product_info, widget_info, widget_names, prediction_date)
    order_features = generate_order_features(actions, product_info, prediction_date)
    search_features = generate_search_features(searches, widget_names, prediction_date)
    interaction_features = generate_interaction_features(actions, product_info)
    time_features = generate_time_features(actions, end_date)
    #svd_features = generate_svd_features(actions)
    
    # Собираем все признаки вместе
    #all_features = action_features + [order_features, interaction_features, time_features] + search_features

    all_features = [order_features, interaction_features, time_features, search_features]

    
    features = all_features[0]
    for df in all_features[1:]:
        features = features.join(df, on='user_id', how='left')
    
    # Заполняем пропуски и создаем взаимодействия признаков
    features = features.fill_null(0)
    
    numeric_cols = [col for col in features.columns if col not in ['user_id'] and features[col].dtype in [pl.Float64, pl.Int64]]
    for col1, col2 in combinations(numeric_cols, 2):
        if f'{col1}_x_{col2}' not in features.columns:
            features = features.with_columns([
                (pl.col(col1) * pl.col(col2)).alias(f'{col1}_x_{col2}'),
                (pl.col(col1) + pl.col(col2)).alias(f'{col1}_plus_{col2}'),
                (pl.col(col1) / (pl.col(col2) + 1e-6)).alias(f'{col1}_div_{col2}'),
            ])
    
    return features




#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




def get_actions_aggs(
    actions_history, 
    product_info, 
    end_date, 
    reference_date, 
    lookback_days=30*4,
    search_history=None
):
    """Генерация агрегатов по действиям пользователей с учетом кластеров поиска."""
    actions_aggs = {}
    actions_id_to_suf = {
        1: "click", 
        2: "favorite",
        3: "order",
        5: "to_cart",
    }
    
    all_aggs = []
    numeric_features = []
    
    # Получаем основной кластер поиска для каждого пользователя (если есть search_history)
    main_search_cluster = None
    if search_history is not None:
        main_search_cluster = (
            search_history
            .filter(pl.col('timestamp').dt.date() <= end_date)
            .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=lookback_days))
            .group_by('user_id')
            .agg(pl.col('cluster').mode().first().alias('main_search_cluster'))
        )
    
    for id_, suf in actions_id_to_suf.items():
        # Базовые агрегаты за весь период (lookback_days)
        base_query = (
            actions_history
            .filter(pl.col('timestamp').dt.date() <= end_date)
            .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=lookback_days))
            .filter(pl.col('action_type_id') == id_)
            .join(
                product_info.select('product_id', 'discount_price', 'cluster'),
                on='product_id',
            )
        )
        
        # Если есть данные о поисках, добавляем информацию о основном кластере поиска
        if main_search_cluster is not None:
            base_query = base_query.join(main_search_cluster, on='user_id', how='left')
        
        aggs = (
            base_query
            .group_by('user_id')
            .agg(
                # Базовые агрегаты
                pl.count('product_id').cast(pl.Int32).alias(f'num_products_{suf}'),
                pl.sum('discount_price').cast(pl.Float32).alias(f'sum_discount_price_{suf}'),
                pl.max('discount_price').cast(pl.Float32).alias(f'max_discount_price_{suf}'),
                pl.max('timestamp').alias(f'last_{suf}_time'),
                pl.min('timestamp').alias(f'first_{suf}_time'),
                pl.col('cluster').n_unique().alias(f'num_{suf}_clusters'),
                pl.col('cluster').mode().first().alias(f'main_{suf}_cluster'),
                (pl.col('cluster').value_counts().struct.field('count').max() / pl.count()).alias(f'{suf}_cluster_concentration'),
                
                # Новые агрегаты по кластерам (если есть данные о поисках)
                *([
                    # Доля действий в основном кластере поиска
                    (pl.col('cluster').filter(pl.col('cluster') == pl.col('main_search_cluster')).count() / 
                    pl.count().alias(f'{suf}_in_main_search_cluster_ratio')),
                    
                    # Количество кластеров действий, которые совпадают с кластерами поиска
                    pl.col('cluster').filter(
                        pl.col('cluster').is_in(
                            search_history.filter(pl.col('user_id') == pl.col('user_id')).select('cluster')
                    ).n_unique().alias(f'num_{suf}_clusters_matching_search')),
                    
                    # Средняя цена в основном кластере поиска vs других кластерах
                    (pl.col('discount_price').filter(pl.col('cluster') == pl.col('main_search_cluster')).mean() - 
                     pl.col('discount_price').filter(pl.col('cluster') != pl.col('main_search_cluster')).mean()
                    ).alias(f'{suf}_price_diff_main_search_cluster'),
                    
                    # Время между поиском и действием в том же кластере (только для действий после поиска)
                    ((pl.col('timestamp') - 
                      search_history.filter(
                          (pl.col('user_id') == pl.col('user_id')) & 
                          (pl.col('cluster') == pl.col('cluster'))
                      ).select(pl.col('timestamp').min())
                     ).dt.total_days().mean().alias(f'mean_days_between_search_and_{suf}_same_cluster'))
                ] if main_search_cluster is not None else []
                )
            )
            .with_columns([
                (pl.lit(reference_date) - pl.col(f'last_{suf}_time'))
                .dt.total_days()
                .cast(pl.Int32)
                .alias(f'days_since_last_{suf}'),
                
                (pl.lit(reference_date) - pl.col(f'first_{suf}_time'))
                .dt.total_days()
                .cast(pl.Int32)
                .alias(f'days_since_first_{suf}'),
            ])
        )
        
        # Добавляем агрегаты за разные временные окна для кликов и заказов
        if id_ == 1:  # clicks
            for i, days in enumerate([30, 60, 90], 1):
                window_aggs = (
                    actions_history
                    .filter(pl.col('timestamp').dt.date() <= end_date)
                    .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=days))
                    .filter(pl.col('action_type_id') == id_)
                    .group_by('user_id')
                    .agg(
                        pl.count('product_id').cast(pl.Int32).alias(f'num_products_{suf}_last_{i}_month'),
                    )
                )
                aggs = aggs.join(window_aggs, on='user_id', how='left')
                numeric_features.append(f'num_products_{suf}_last_{i}_month')
                
        elif id_ == 3:  # orders
            for i, days in enumerate([30, 60, 90], 1):
                window_aggs = (
                    actions_history
                    .filter(pl.col('timestamp').dt.date() <= end_date)
                    .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=days))
                    .filter(pl.col('action_type_id') == id_)
                    .join(
                        product_info.select('product_id', 'discount_price'),
                        on='product_id',
                    )
                    .group_by('user_id')
                    .agg(
                        pl.sum('discount_price').cast(pl.Float32).alias(f'sum_discount_price_{suf}_last_{i}_month'),
                    )
                )
                aggs = aggs.join(window_aggs, on='user_id', how='left')
                numeric_features.append(f'sum_discount_price_{suf}_last_{i}_month')
        
        # Добавляем фичи в список числовых фич
        base_features = [
            f'num_products_{suf}',
            f'sum_discount_price_{suf}', 
            f'max_discount_price_{suf}',
            f'days_since_last_{suf}',
            f'days_since_first_{suf}',
            f'num_{suf}_clusters',
            f'main_{suf}_cluster',
            f'{suf}_cluster_concentration',
        ]
        
        if main_search_cluster is not None:
            base_features.extend([
                f'{suf}_in_main_search_cluster_ratio',
                f'num_{suf}_clusters_matching_search',
                f'{suf}_price_diff_main_search_cluster',
                f'mean_days_between_search_and_{suf}_same_cluster',
            ])
        
        numeric_features.extend(base_features)
        
        actions_aggs[id_] = aggs
        all_aggs.append(aggs)

        print(f"actions {id_} {suf} finished")
    
    # Объединяем все агрегации
    combined = all_aggs[0]
    for i, agg in enumerate(all_aggs[1:], 1):
        combined = combined.join(agg, on='user_id', how='left', suffix=f"_{i}")
    
    return actions_aggs, combined, numeric_features


def get_search_aggs(search_history, end_date, reference_date, lookback_days=30*5):
    """Генерация агрегатов по поисковым запросам."""
    id_ = 4
    suf = 'search'
    
    # Топ-3 кластеров
    cluster_counts = (
        search_history
        .filter(pl.col('action_type_id') == id_)
        .filter(pl.col('timestamp').dt.date() <= end_date)
        .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=lookback_days))
        .group_by('user_id')
        .agg(
            pl.col('cluster').value_counts().alias('cluster_counts')
        )
        .explode('cluster_counts')
        .with_columns(
            pl.col('cluster_counts').struct.field('cluster').alias('cluster_name'),
            pl.col('cluster_counts').struct.field('count').alias('cluster_count')
        )
        .group_by('user_id')
        .agg(
            pl.col('cluster_name').sort_by('cluster_count', descending=True).head(3).alias('top3_clusters'),
            pl.col('cluster_count').sort(descending=True).head(3).alias('top3_counts')
        )
    )
    
    # Агрегаты по поискам
    search_aggs = (
        search_history
        .filter(pl.col('action_type_id') == id_)
        .filter(pl.col('timestamp').dt.date() <= end_date)
        .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=lookback_days))
        .group_by('user_id')
        .agg(
            pl.count('search_query').cast(pl.Int32).alias(f'num_{suf}'),
            pl.col('search_query').n_unique().alias(f'unique_{suf}_queries'),
            
            pl.col('search_query')
                .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=30))
                .count()
                .cast(pl.Int32)
                .alias(f'num_{suf}_last_month'),
            
            pl.col('search_query')
                .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=7))
                .count()
                .cast(pl.Int32)
                .alias(f'num_{suf}_last_week'),

            (pl.count() / (pl.max('timestamp') - pl.min('timestamp')).dt.total_days()).alias(f'{suf}_daily_rate'),

            pl.col('cluster').n_unique().alias(f'num_{suf}_clusters'),
            pl.col('cluster').mode().first().alias(f'main_{suf}_cluster'),
            
            pl.col('cluster')
                .filter(pl.col('timestamp').dt.date() >= end_date - timedelta(days=30))
                .mode().first()
                .alias(f'recent_{suf}_cluster'),

            (pl.col('cluster').value_counts().struct.field('count').max() / pl.col('cluster').count()).alias(f'{suf}_cluster_concentration'),
            
            (-(pl.col('cluster').value_counts().struct.field('count') / pl.col('cluster').count()).log()
                * (pl.col('cluster').value_counts().struct.field('count') / pl.col('cluster').count())
                .sum()).alias(f'{suf}_cluster_entropy'),
            
            pl.col('cluster').diff().fill_null(0).abs().sum().alias(f'{suf}_cluster_switches'),
            
            ((pl.col('cluster').count() - pl.col('cluster').n_unique()) / pl.col('cluster').count())
                .alias(f'{suf}_cluster_stability'),
            
            (pl.col('timestamp')
                .filter(pl.col('cluster') == pl.col('cluster').mode().first())
                .count() / pl.col('timestamp').count())
                .alias(f'main_{suf}_cluster_time_ratio'),

            pl.col('timestamp').filter(pl.col('cluster').diff().fill_null(0) != 0)
                .diff()
                .dt.total_days()
                .mean()
                .alias(f'{suf}_mean_cluster_switch_days'),

            pl.col('search_query').str.len_chars().mean().alias(f'{suf}_mean_query_len'),
            
            (pl.col('search_query').str.len_chars()
                .filter(pl.col('cluster') == pl.col('cluster').mode().first()).mean() - 
                pl.col('search_query').str.len_chars()
                    .filter(pl.col('cluster') != pl.col('cluster').mode().first()).mean())
                    .alias(f'{suf}_main_cluster_query_len_diff'),

            pl.max('timestamp').alias(f'last_{suf}_time'),
            pl.min('timestamp').alias(f'first_{suf}_time'),
        )
        .join(cluster_counts, on='user_id', how='left')
        .with_columns([
            (pl.lit(reference_date) - pl.col(f'last_{suf}_time'))
                .dt.total_days()
                .cast(pl.Int32)
                .alias(f'days_since_last_{suf}'),

            (pl.lit(reference_date) - pl.col(f'first_{suf}_time'))
                .dt.total_days()
                .cast(pl.Int32)
                .alias(f'days_since_first_{suf}'),
        ])
        .select(
            'user_id',
            f'num_{suf}',
            f'unique_{suf}_queries',
            f'num_{suf}_last_month',
            f'num_{suf}_last_week',
            f'{suf}_daily_rate',
            f'num_{suf}_clusters',
            f'main_{suf}_cluster',
            pl.col('top3_clusters').alias(f'top3_{suf}_clusters'),
            pl.col('top3_counts').alias(f'top3_{suf}_counts'),
            f'recent_{suf}_cluster',
            f'{suf}_cluster_concentration',
            f'{suf}_cluster_entropy',
            f'{suf}_cluster_switches',
            f'{suf}_cluster_stability',
            f'main_{suf}_cluster_time_ratio',
            f'{suf}_mean_cluster_switch_days',
            f'{suf}_mean_query_len',
            f'{suf}_main_cluster_query_len_diff',
            f'days_since_last_{suf}',
            f'days_since_first_{suf}',
            f'last_{suf}_time',
            f'first_{suf}_time',
        )
    )
    
    return {id_: search_aggs}

def build_features(base_df, actions_aggs, search_aggs):
    """Построение финального датафрейма с признаками."""
    df_main = base_df
    for _, actions_aggs_df in actions_aggs.items():
        df_main = df_main.join(actions_aggs_df, on='user_id', how='left')
    
    for _, search_aggs_df in search_aggs.items():
        df_main = df_main.join(search_aggs_df, on='user_id', how='left')
    
    return df_main
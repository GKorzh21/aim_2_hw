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

def add_knn_features(train, test, features, target, n_neighbors=5):
    train_filled = train.copy()
    test_filled = test.copy()
    
    # Определяем колонки для заполнения (кроме target и user_id)
    cols_to_fill = [col for col in features if col not in ['user_id', 'target']]
    
    # Функция для умного заполнения
    def smart_fill(df):
        # Заполняем пропуски в строках, где есть num_products_to_cart
        mask = df['num_products_click'].notna()
        
        # Для числовых колонок
        num_cols = [col for col in cols_to_fill if df[col].dtype in ['int64', 'float64']]
        df.loc[mask, num_cols] = df.loc[mask, num_cols].fillna({
            # Для счетчиков - 0
            **{col: 0 for col in num_cols if 'num_' in col},
            # Для цен - медиана
            **{col: df[col].median() for col in num_cols if 'price' in col},
            # Для дней - максимальное значение + 1 (как "очень давно")
            **{col: df[col].max() + 1 for col in num_cols if 'days_' in col}
        })
        
        # Для временных меток - заполняем минимальной датой
        time_cols = [col for col in cols_to_fill if 'time' in col]
        min_date = pd.to_datetime('2024-01-01')  # Очень старая дата
        df.loc[mask, time_cols] = df.loc[mask, time_cols].fillna(min_date)
        
        return df
    
    # Применяем умное заполнение
    train_filled = smart_fill(train_filled)
    test_filled = smart_fill(test_filled)

    display(train_filled)
    
    # Добавляем метки полных строк (после заполнения)
    train_filled['_complete_row'] = ~train_filled[features].isna().any(axis=1)
    test_filled['_complete_row'] = ~test_filled[features].isna().any(axis=1)
    
    print(f"Тренировочных строк без пропусков: {train_filled['_complete_row'].sum()}/{len(train_filled)}")
    print(f"Тестовых строк без пропусков: {test_filled['_complete_row'].sum()}/{len(test_filled)}")
    
    # Обучаем KNN только на полных строках
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(train_filled.loc[train_filled['_complete_row'], features])
    
    def get_stats(X, X_filled):
        # Инициализируем результат с -1
        stats = pd.DataFrame({
            'knn_mean_target': -1,
            'knn_std_target': -1,
            'knn_median_dist': -1,
            'knn_min_dist': -1,
            'knn_max_dist': -1
        }, index=X.index)
        
        # Вычисляем статистики только для полных строк
        complete_rows = X_filled['_complete_row']
        if complete_rows.any():
            distances, indices = knn.kneighbors(X.loc[complete_rows, features])
            
            stats.loc[complete_rows, 'knn_mean_target'] = train.iloc[indices.flatten()][target].values.reshape(indices.shape).mean(axis=1)
            stats.loc[complete_rows, 'knn_std_target'] = train.iloc[indices.flatten()][target].values.reshape(indices.shape).std(axis=1)
            stats.loc[complete_rows, 'knn_median_dist'] = np.median(distances, axis=1)
            stats.loc[complete_rows, 'knn_min_dist'] = np.min(distances, axis=1)
            stats.loc[complete_rows, 'knn_max_dist'] = np.max(distances, axis=1)
        
        return stats
    
    # Добавляем статистики
    train_stats = get_stats(train_filled, train_filled)
    test_stats = get_stats(test_filled, test_filled)
    
    # Объединяем с исходными данными
    train_result = pd.concat([train_filled, train_stats], axis=1)
    test_result = pd.concat([test_filled, test_stats], axis=1)
    
    # Удаляем временные колонки
    train_result.drop(columns=['_complete_row'], inplace=True)
    test_result.drop(columns=['_complete_row'], inplace=True)
    
    return train_result, test_result
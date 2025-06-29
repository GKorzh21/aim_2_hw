import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib as mpl
from matplotlib.patches import Patch
from typing import Optional, Literal

#TODO: Добавить предварительное сэмплирование, если на вход подан слишком большой датасет.
def plot_shap_beeswarm(
        df: pd.DataFrame,
        feature_names: Optional[list[str]] = None,
        shap_values: object = None,
        max_features: int = 20,
        cat_feature_threshold: float = 0.01,
        figsize: tuple[int, int] = (10, 6),
        n_samples: int = 1_000,
        palette_num: str | list = "coolwarm",
        palette_cat: str | list = "tab20",
        zero_line_color: str = "red",
        title: Optional[str] = None,
        rare_policy: Literal["combine", "drop"] = "combine",
        random_state: int = 8792,
        alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Строит расширенный beeswarm-график SHAP значений с поддержкой категориальных признаков,
    отображением пропусков (NaN), цветовой нормализацией и контролируемой выборкой.
    Все уникальные значения категориальных признаков становятся отдельными признаками с именем в формате
    "родитель#категория".
    Для них строятся boxplot c усами по всему диапазону (min, max).
    Для категорий родителя вводится порог частоты для определения редких категорий (cat_feature_threshold).
    Если в параметре `rare_policy`='drop', то редкие категории удаляются, если 'combine', то создается
    признак "родитель#COMBINED_RARE".
    Все признаки, полученные от одного родителя, красятся в одинаковый цвет.

    :param df: Оригинальный DataFrame с признаками.
    :param feature_names: Список признаков для отображения. Если None — используются все.
    :param shap_values: SHAP-значения, соответствующие df по размеру (может быть DataFrame или ndarray).
    :param max_features: Максимальное количество признаков (или категорий) для отображения.
    :param cat_feature_threshold: Порог для определения редких категорий (по доле).
    :param figsize: Размер графика в дюймах (ширина, высота).
    :param n_samples: Количество наблюдений для отображения (сэмплируется).
    :param palette_num: Цветовая палитра или colormap для числовых признаков.
    :param palette_cat: Цветовая палитра для категориальных признаков.
    :param zero_line_color: Цвет вертикальной линии SHAP=0.
    :param title: Заголовок графика (опционально). Если None, то не отображается.
    :param rare_policy: Политика обработки редких категорий: 'combine' — объединить, 'drop' — удалить.
    :param random_state: Начальное значение генератора случайных чисел (для воспроизводимости).
    :param alpha: Прозрачность точек и boxplot (от 0 до 1).
    :return: DataFrame, содержащий точки для построения графика (после обработки и сэмплирования).
    """

    rng = np.random.default_rng(random_state)

    # Преобразуем SHAP-массив в DataFrame (если передан np.ndarray)
    if isinstance(shap_values, np.ndarray):
        shap_df = pd.DataFrame(shap_values, index=df.index, columns=df.columns)
    else:
        shap_df = shap_values.copy()

    # Проверка на соответствие форм и корректность параметров
    if shap_df.shape != df.shape:
        raise ValueError("Shape mismatch between df and shap_values.")
    if not (0 < cat_feature_threshold < 1):
        raise ValueError("cat_feature_threshold must be between 0 and 1.")
    if rare_policy not in {"combine", "drop"}:
        raise ValueError("rare_policy must be 'combine' or 'drop'.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")

    # Сужаем датафрейм до выбранных признаков
    if feature_names:
        parents = [f for f in feature_names if f in df.columns]
    else:
        parents = list(df.columns)
    df = df[parents]
    shap_df = shap_df[parents]

    # Разделяем признаки на числовые и категориальные
    num_cols = [f for f in parents if df[f].dtype.name not in {"object","category"}]
    cat_cols = [f for f in parents if f not in num_cols]

    # Расширяем категориальные признаки: делаем отдельные колонки SHAP-значений на каждую категорию
    expanded = {}
    for f in cat_cols:
        series = df[f].astype(object)
        freq = series.value_counts(dropna=False, normalize=True)
        rare = freq[freq < cat_feature_threshold].index
        proc = series.where(~series.isin(rare), "COMBINED_RARE") if rare_policy=="combine" else series[~series.isin(rare)]
        proc = proc.fillna("NaN")
        shap_col = shap_df[f]
        for cat in proc.unique():
            expanded[f"{f}#{cat}"] = shap_col.where(proc==cat)

    # Для числовых признаков просто копируем SHAP значения
    for f in num_cols:
        expanded[f] = shap_df[f]
    shap_expanded = pd.DataFrame(expanded)

    # Сортировка признаков по среднему абсолютному SHAP влиянию
    mean_abs = shap_expanded.abs().mean().sort_values(ascending=False)
    ordered = list(mean_abs.index)[:max_features]

    # Подготовка данных для отображения: выборка сэмплов и нормализация
    records = []
    norm_map = {f: df[f].rank(pct=True) for f in num_cols} # приведение к равномерному распределению. TODO: потом сделаю умнее
    for feat in ordered:
        vals = shap_expanded[feat].dropna()
        idx = rng.choice(vals.index, min(len(vals), n_samples), replace=False)
        is_cat = feat not in num_cols
        for i in idx:
            base_feat = feat.split("#", 1)[0] if is_cat else feat
            orig_val = df[base_feat].loc[i]
            norm_val = np.nan if is_cat else norm_map[feat].loc[i]
            is_nan = pd.isna(orig_val)
            records.append({
                'feature': feat,
                'shap': vals.loc[i],
                'norm_val': norm_val,
                'is_cat': is_cat,
                'is_nan': is_nan,
                'cat_value': (feat.split('#', 1)[1] if is_cat else np.nan)
            })
    plot_df = pd.DataFrame(records)
    plot_df['feature'] = pd.Categorical(plot_df['feature'], categories=ordered, ordered=True)

    # Настройка цвета: градиент для числовых, цветовая палитра для категориальных
    num_norm = Normalize(vmin=0, vmax=1)
    cmap_num = mpl.colormaps.get_cmap(palette_num) if isinstance(palette_num,str) else palette_num
    parent_cat_features = sorted({c.split('#', 1)[0] for c in ordered if '#' in c})
    cat_palette_used = sns.color_palette(palette_cat, len(parent_cat_features)) if isinstance(palette_cat,str) else palette_cat
    parent_color_map = dict(zip(parent_cat_features, cat_palette_used))

    # Построение графика
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    ax = fig.add_axes([0.05,0.15,0.6,0.75])

    # Boxplot'ы для категориальных признаков
    for i, feat in enumerate(ordered):
        if feat not in num_cols:
            dfc = plot_df[plot_df.feature==feat]
            parent = feat.split('#', 1)[0]
            vals = dfc.shap.values
            bp = ax.boxplot(
                vals,
                positions=[i],
                vert=False,
                widths=0.6,
                patch_artist=True,
                whis=(0, 100),
                zorder=2
            )
            for box in bp['boxes']:
                box.set_facecolor(parent_color_map[parent])
                box.set_alpha(alpha)
            for whisker in bp['whiskers']:
                whisker.set_color(parent_color_map[parent])
            for cap in bp['caps']:
                cap.set_color(parent_color_map[parent])
            for median in bp['medians']:
                median.set_color('black')

    # Scatter точки для числовых признаков: джиттер + отдельный рендер NaN
    for i, feat in enumerate(ordered):
        if feat in num_cols:
            # Точки с известными значениями
            df_num = plot_df[(plot_df.feature == feat) & (~plot_df.is_nan)]
            if not df_num.empty:
                y_jitter = i + (rng.random(len(df_num)) - 0.5) * 0.4
                ax.scatter(
                    df_num.shap,
                    y_jitter,
                    c=[cmap_num(num_norm(v)) for v in df_num.norm_val],
                    s=20,
                    alpha=alpha,
                    linewidths=0,
                    zorder=3
                )
            # Черные точки для NaN (без джиттера)
            df_nan_num = plot_df[(plot_df.feature == feat) & (plot_df.is_nan)]
            if not df_nan_num.empty:
                ax.scatter(
                    df_nan_num.shap,
                    [i] * len(df_nan_num),
                    c='black',
                    s=20,
                    alpha=alpha,
                    linewidths=0,
                    zorder=4
                )

    # Вертикальная линия SHAP = 0
    ax.axvline(0, color=zero_line_color, linewidth=1.5, zorder=1)

    # Подписи по оси Y
    y_labels=[]
    for feat in ordered:
        if '#' in feat:
            pre,cat=feat.split('#',1)
            safe_pre=pre.replace('_', '\\_').replace(' ', '\\ ')
            safe_cat=cat.replace('_', '\\_').replace(' ', '\\ ')
            y_labels.append(r"$\mathbf{"+safe_pre+r"}\#\mathrm{"+safe_cat+r"}$")
        else:
            sf=feat.replace('_','\\_').replace(' ','\\ ')
            y_labels.append(r"$\mathbf{"+sf+r"}$")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP values')
    ax.set_ylabel('Feature')
    ax.grid(axis='x', linestyle='--', alpha=0)
    if title:
        ax.set_title(title, fontsize=14)

    # Цветовая шкала для числовых признаков
    sm=mpl.cm.ScalarMappable(cmap=cmap_num, norm=num_norm)
    sm.set_array([])
    cax=fig.add_axes([0.68,0.15,0.02,0.75])
    cbar=fig.colorbar(sm, cax=cax)

    # Легенда для категорий и NaN
    legend_elems=[Patch(facecolor='black', label='NaN')]+[Patch(facecolor=parent_color_map[p], label=p) for p in parent_cat_features]
    fig.legend(
        handles=legend_elems,
        title='Categories',
        bbox_to_anchor=(0.75,0.5),
        loc='center left',
        frameon=False
    )

    return plot_df

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics

APP_DIR = Path(__file__).parent          

@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')

@st.cache_data
def load_test():
    df = pd.read_csv(APP_DIR / 'test.csv')
    return df.drop(columns='target'), df['target']

@st.cache_data
def test_proba(_model, X):              
    return _model.predict_proba(X)[:, 1]

@st.cache_data
def curves(proba, y):
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    prec, rec, _ = metrics.precision_recall_curve(y, proba)
    return fpr, tpr, prec, rec

model = load_model()
X_test, y_test = load_test()
FEATURES = list(X_test.columns)          
default_t = json.loads((APP_DIR / 'config.json').read_text())['threshold']

st.title('Рекомендации вин')
threshold = st.sidebar.slider('Порог', 0.0, 1.0, float(default_t), 0.01)

tab1, tab2, tab3, tab4 = st.tabs(['Одно вино', 'Модельные метрики',
                                  'Пакетное предсказание', 'Что выучила модель'])
with tab1:
    st.subheader('Оценка одного вина')

    values = {}
    cols = st.columns(3)                      
    for i, col in enumerate(FEATURES):
        with cols[i % 3]:
            if col == 'is_red':
                color = st.radio('Цвет', ['белое', 'красное'], horizontal=True)
                values[col] = 1 if color == 'красное' else 0
            else:
                lo, hi = float(X_test[col].min()), float(X_test[col].max())
                values[col] = st.number_input(
                    col, min_value=lo, max_value=hi,
                    value=float(X_test[col].median()),
                    step=(hi - lo) / 100, format='%.4f')

    row = pd.DataFrame([values])[FEATURES]   

    proba = model.predict_proba(row)[0, 1]
    decision = proba >= threshold

    st.metric('Вероятность, что вино хорошее', f'{proba:.1%}')
    if decision:
        st.success(f'Решение модели: рекомендовать (порог {threshold:.2f})')
    else:
        st.warning(f'Решение модели: не рекомендовать (порог {threshold:.2f})')

with tab2:
    st.subheader('Модельные метрики на тестовой выборке')

    proba = test_proba(model, X_test)            # из кэша, от порога не зависит
    y_true = y_test.to_numpy()
    pred = (proba >= threshold).astype(int)      # меняется вместе со слайдером

    # --- матрица ошибок ---
    cm = metrics.confusion_matrix(y_true, pred)  # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    # --- метрики ---
    prec_t = metrics.precision_score(y_true, pred, zero_division=0)
    rec_t = metrics.recall_score(y_true, pred)
    f1_t = metrics.f1_score(y_true, pred)

    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f'{prec_t:.3f}')
    c2.metric('Recall', f'{rec_t:.3f}')
    c3.metric('F1', f'{f1_t:.3f}')

    # --- рисунок матрицы ошибок ---
    fig_cm, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['не рекомендуем', 'рекомендуем'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['плохое', 'хорошее'][::-1][::-1])
    ax.set_xlabel('Решение модели'); ax.set_ylabel('На самом деле')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha='center', va='center',
                color='white' if v > cm.max() / 2 else 'black', fontsize=14)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # --- ROC и PR с отметкой текущего порога ---
    fpr, tpr, prec, rec = curves(proba, y_true)
    cur_fpr = fp / (fp + tn)
    cur_tpr = tp / (tp + fn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_true, proba):.3f}')
    ax1.plot([0, 1], [0, 1], '--', color='gray')
    ax1.scatter(cur_fpr, cur_tpr, color='red', s=80, zorder=3, label=f'порог {threshold:.2f}')
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.set_title('ROC-кривая'); ax1.legend()

    ax2.plot(rec, prec, label=f'PR-AUC = {metrics.average_precision_score(y_true, proba):.3f}')
    ax2.axhline(y_true.mean(), linestyle=':', color='gray', label='случайная модель')
    ax2.scatter(rec_t, prec_t, color='red', s=80, zorder=3, label=f'порог {threshold:.2f}')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('PR-кривая'); ax2.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        '**Что делает порог.** Если порог увеличить, модель рекомендует только вина, '
        'в которых очень уверена: в подборке меньше посредственных вин, '
        'но больше хороших вин остаётся за её пределами. '
        'Если порог уменьшить, подборка становится шире и находит больше хороших вин, '
        'но в неё попадает больше посредственных.'
    )

with tab3:
    st.subheader('Пакетное предсказание')
    st.write('Загрузите CSV с признаками вин (без столбца `target`). '
             f'Решение принимается при текущем пороге: **{threshold:.2f}**.')

    file = st.file_uploader('CSV-файл с признаками вин', type='csv')

    if file is not None:
        try:
            df_in = pd.read_csv(file)
        except Exception as e:
            st.error(f'Не удалось прочитать файл как CSV: {e}')
            st.stop()

        missing = [c for c in FEATURES if c not in df_in.columns]
        if missing:
            st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            st.stop()

        X_in = df_in[FEATURES].apply(pd.to_numeric, errors='coerce')
        if X_in.isna().any().any():
            bad = X_in.columns[X_in.isna().any()].tolist()
            st.error('В столбцах есть пропуски или нечисловые значения: ' + ', '.join(bad))
            st.stop()

        proba_in = model.predict_proba(X_in)[:, 1]
        result = df_in.copy()
        result['вероятность'] = proba_in.round(4)
        result['решение'] = np.where(proba_in >= threshold, 'рекомендовать', 'не рекомендовать')

        st.success(f'Рекомендовано {(proba_in >= threshold).sum()} из {len(result)} вин.')
        st.dataframe(result, use_container_width=True)
        st.download_button(
            'Скачать результат (CSV)',
            data=result.to_csv(index=False).encode('utf-8-sig'),
            file_name='wine_predictions.csv',
            mime='text/csv',
        )

with tab4:
    st.subheader('Что выучила модель')

    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()

    colors = ['tab:red' if c < 0 else 'tab:green' for c in coefs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Коэффициент (признаки масштабированы)')
    ax.set_title('Вклад признаков в вероятность «хорошее вино»')
    st.pyplot(fig)
    plt.close(fig)

    top_for = coefs.sort_values(ascending=False).head(3).index.tolist()
    top_against = coefs.sort_values().head(3).index.tolist()
    st.write('**Сильнее всего «за» хорошее вино:** ' + ', '.join(top_for))
    st.write('**Сильнее всего «против»:** ' + ', '.join(top_against))

    st.caption(
        'Зелёные столбцы повышают вероятность того, что вино хорошее, красные понижают её. '
        'Признаки приведены к одному масштабу, поэтому длины столбцов сравнимы между собой. '
        'Это связи, которые модель нашла в данных, а не причины: например, коррелированные '
        'признаки (плотность, сахар, спирт) делят между собой общий вклад.'
    )
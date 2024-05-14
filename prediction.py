from sklearn.metrics import r2_score, mean_squared_error
from streamlit_option_menu import option_menu
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures


@st.cache_data
def plot_energy_prices(df):
    plt.figure(figsize=(15, 8))
    for i in range(len(df) // 24):
        colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
        daily_data = df.iloc[i * 24: (i + 1) * 24]
        date = pd.to_datetime(daily_data["Время"]).dt.strftime('%Y-%m-%d').iloc[0]
        plt.plot(
            daily_data['ДаммиЧасы'], daily_data['Цена'],
            alpha=0.5, color=colors[i % len(colors)],
            marker='.', markersize=12, label=date
        )

    plt.title('Цены на электроэнергию по часам', fontsize=16)
    plt.xlabel('Час', fontsize=14)
    plt.ylabel('Цена', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 24))
    plt.grid(True)
    plt.legend(loc='lower right')
    st.pyplot(plt)


def create_plot(df, feature):
    fig = go.Figure(go.Scatter(
        x=df['Время'],
        y=df[feature],
        line=dict(color='black', width=3)
    ))

    fig.update_layout(
        title=f"Распределение {feature}",
        template="plotly_white",
        xaxis_title='Время',
        yaxis_title=feature,
        font=dict(size=12),
        xaxis=dict(
            tickangle=45,
            type='category',
            ticklabelstep=3
        )
    )
    return fig


@st.cache_data
def plot_forecast_real_data(full_df, y_train, y_pred_poly):
    real_data_trace = go.Scatter(
        x=full_df.Время,
        y=y_train.Цена,
        mode='lines',
        name='Исторические данные',
        line=dict(width=4)
    )

    forecast_trace = go.Scatter(
        x=full_df.Время,
        y=y_pred_poly,
        mode='lines',
        name='Прогноз',
        line=dict(color='black', width=4, dash='dash')
    )

    fig = go.Figure([real_data_trace, forecast_trace])
    fig.update_layout(
        title='Прогноз на исторические данные',
        template="plotly_white",
        xaxis_title='Время',
        yaxis_title='Цена',
        font=dict(size=14),
        xaxis=dict(
            tickangle=45,
            type='category',
            ticklabelstep=3
        ))
    return fig


@st.cache_data
def plot_forecast_test_data(full_df, test_data, y_pred_poly):
    hist_data_trace = go.Scatter(
        x=full_df.Время,
        y=full_df.Цена,
        mode='lines',
        name='Исторические данные',
        line=dict(width=3)
    )
    real_data_trace = go.Scatter(
        x=test_data.Время,
        y=test_data.Цена,
        mode='lines',
        name='Реальные данные',
        line=dict(color='lightskyblue', width=4)
    )

    forecast_trace = go.Scatter(
        x=test_data.Время,
        y=y_pred_poly,
        mode='lines',
        name='Прогноз',
        line=dict(color='black', width=4, dash='dash')
    )

    fig = go.Figure(
        [hist_data_trace, real_data_trace, forecast_trace]
    )
    fig.update_layout(
        title='Прогноз на 29е февраля',
        template="plotly_white",
        xaxis_title='Дата',
        yaxis_title='Цена',
        font=dict(size=14),
        xaxis=dict(
            tickangle=45,
            type='category',
        )
    )
    return fig


@st.cache_data
def plot_ape_mape(actual, pred):
    ape_values = np.abs((np.array(pred) - np.array(actual)) / np.array(actual)) * 100
    mape = np.mean(ape_values)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(range(len(pred))),
        y=ape_values,
        name='APE',
        marker_color='blue'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(pred))),
        y=[mape] * len(pred),
        mode='lines',
        name=f'MAPE = {mape:.3f}%',
        marker_color='red',
        line=dict(dash='dash', width=5)
    ))

    fig.update_layout(
        title='График APE с MAPE',
        template="plotly_white",
        xaxis_title='Номер точки данных',
        yaxis_title='APE (%)',
        legend_title="Легенда",
        font=dict(
            size=14,
        ),
    )
    return fig

def decode_inputs(inputs, numerical_features, hours_features):
    inputs_data = {}
    for feat in numerical_features:
        inputs_data[feat] = float(inputs[feat])

    for i, hour in enumerate(hours_features):
        inputs_data[hour] = False
        if inputs['ДаммиЧасы'] == hour:
            inputs_data['ДаммиЧасы'] = i
            inputs_data[hour] = True

    inputs_data['ДаммиНомерНедели'] = inputs['ДаммиНомерНедели']
    inputs_data['ДаммиПогодныеУсловия'] = inputs['ДаммиПогодныеУсловия'] != 'Да'
    final_df = pd.DataFrame([inputs_data])
    return final_df


def app(df: pd.DataFrame, current_dir: Path):
    numerical_features     = ['Цена', 'T', 'Доллар', 'Длина ночи', 'Потребление']
    categorical_features   = ['ДаммиЧасы', 'ДаммиНомерНедели', 'ДаммиПогодныеУсловия']
    hours_features         = [f'{hour}:00-{hour + 1}:00' for hour in range(23)] + ['23:00-0:00']
    categorical_features  += hours_features

    st.title("Прогнозирование цен на электроэнергию")
    st.markdown("""
        ## Описание прогнозируемых данных
        Данные для анализа взяты за отопительный период, охватывающий декабрь 2023, январь и февраль 2024 года.
        В этот период потребление электроэнергии возрастает из-за увеличения использования отопительных приборов и освещения.
        Источник данных: [time2save](https://time2save.ru/baza-tarifov-na-elektroenergiyu)
        Погодные данные: [rp5.ru](https://rp5.ru/Архив_погоды_в_Москве_(ВДНХ))
        - **Дискретизация временного ряда:** Часовая дискретизация данных, предоставленных источником.
        - **Файл данных:** Данные, взятые с сайта time2save, содержат информацию о ценах на электроэнергию, которые использовались для анализа.
        - **Метод анализа:** Использование полинома второй степени для моделирования зависимостей.
        - **Исторический период:** Анализ данных за декабрь 2023, январь и февраль 2024 года.
        """)
    # Вставка картинки (предварительно загруженной на сервер)
    st.image(str(current_dir / 'images' / 'calendar.jpg'), width=600, caption='Календарь анализируемого периода')

    st.markdown(r"""
        ## Выбор горизонта прогноза.
        Горизонт планирования в 1 день (24 часа) даст возможность более точно спрогнозировать цену и вывести более точную ошибку прогноза.

        ## Методология расчета точности прогноза
        Для оценки точности прогнозов используется средняя абсолютная процентная ошибка (MAPE).
        Эта метрика измеряет среднее отклонение прогнозируемых значений от фактических значений и выражается в процентах.
        Формула для вычисления средней абсолютной процентной ошибки (MAPE - Mean Absolute Percentage Error) выглядит следующим образом:
        $$
        MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{Y_i - F_i}{Y_i} \right| \times 100
        $$

        Где:
        - $ n $ - количество наблюдений (дней, в данном случае)
        - $ Y_i $ - фактическое значение в момент времени $ i $
        - $ F_i $ - прогнозное значение в момент времени $ i $

        Определим требуемую точность наших прогнозов. Мы видим, что минимальное значение составляет 2890.9, максимальное - 4416.01, и среднее значение - 3955.5618. Основываясь на этих данных, мы хотим определить, насколько точными должны быть наши прогнозы.
    """)
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Обоснование требуемого уровня точности прогноз                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # Рассчет требуемой точности прогнозов
    min_value = df['Цена'].min()
    max_value = df['Цена'].max()
    mean_value = df['Цена'].mean()

    # Рассчитываем MAPE на примере минимальной, максимальной и средней цены
    def calculate_mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    actual_values = [min_value, max_value]
    predicted_values = [mean_value, mean_value]
    mape_value = calculate_mape(np.array(actual_values), np.array(predicted_values))

    # Установка допустимой погрешности прогноза
    desired_accuracy = 5  # 5% уровень допустимой погрешности

    # Настройка страницы
    st.markdown("""
        ## Обоснование требуемого уровня точности прогноза
        Чтобы определить, насколько точными должны быть наши прогнозы, мы исследуем разброс фактических значений цен.
        """)
    st.markdown(f"""
        - **Минимальное значение цены:** {min_value:.2f}
        - **Максимальное значение цены:** {max_value:.2f}
        - **Среднее значение цены:** {mean_value:.2f}
        """)
    st.markdown(f"""
        **Рассчитанная MAPE (Средняя абсолютная процентная ошибка):** {mape_value:.2f}%
        """)
    st.markdown(r"""
        Этот показатель MAPE демонстрирует, что среднее абсолютное процентное отклонение между фактическими и прогнозируемыми значениями составляет 23.97%.
        $$
        \mathrm{\Delta}_{Прогноза} = \left\lceil\frac{Δ_MAPE}{5}\right\rceil = \left\lceil\frac{23.95\%}{5}\right\rceil = \left\lceil\frac{4.7937\%}{5}\right\rceil = 5\%
        $$

        Для упрощения и достижения требуемой точности прогноза мы делим на 5 и округляем это значение, в итоге допустимый уровень ошибки для наших прогнозов равен 5%.
        """)

    # Визуализация расчетов
    if st.button("Показать расчеты MAPE"):
        st.write(f"Фактические значения: {actual_values}")
        st.write(f"Прогнозируемые значения: {predicted_values}")
        st.write(f"Расчет MAPE: {mape_value:.2f}%")
        st.write(f"Допустимый уровень ошибки (ΔПрогноза): {desired_accuracy}%")

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Разделение данных                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Разделение данных на обучающие и проверочные наборы")
    train_data = df[:(len(df) // 24 - 1) * 24]
    test_data = df[(len(df) // 24 - 1) * 24:]

    X_train = train_data.drop(['Время', 'Цена'], axis=1)
    X_train = X_train.reindex(sorted(X_train.columns, reverse=True), axis=1)
    y_train = train_data[['Цена']]

    X_test = test_data.drop(['Время', 'Цена'], axis=1)
    X_test = X_test.reindex(sorted(X_test.columns, reverse=True), axis=1)
    y_test = test_data[['Цена']]

    tab1, tab2 = st.tabs(["Тренировочные данные", "Тестовые данные"])

    with tab1:
        st.subheader("Тренировочные данные")
        st.markdown("""
            **Описание:** Тренировочные данные используются для подгонки модели и оценки её параметров. 
            Эти данные получены путем исключения из исходного датасета столбцов с временными метками и целевой переменной 'Цена'.
            """)
        plot_energy_prices(train_data)
        st.markdown("""
            **Данные тренировочного набора (X_train)**.
            Обучающий набор данных содержит информацию о признаках, используемых для обучения модели. 
        """)
        st.dataframe(X_train)
        st.markdown("""
            **Целевая переменная (y_train)**.
            Целевая переменная содержит значения цены, которые модель должна научиться прогнозировать. 
            В качестве целевой переменной для тренировочного набора используются исключительно значения столбца 'Цена'.
        """)
        st.write(y_train.T)

        st.header("Визуализация числовых признаков")
        selected_feature = st.selectbox(
            "Выберите признак",
            numerical_features,
            key="create_histogram_selectbox1"
        )
        st.plotly_chart(create_plot(train_data, selected_feature), use_container_width=True)

    with tab2:
        st.subheader("Тестовые данные")
        st.markdown("""
            **Описание:** Тестовые данные используются для проверки точности модели на данных, которые не участвовали в тренировке.
            Это позволяет оценить, как модель будет работать с новыми, ранее не виденными данными.
            """)
        plot_energy_prices(test_data)
        st.markdown("""
            **Данные тестового набора (X_test)**.
            Тестовый набор данных содержит информацию о признаках, используемых для оценки модели.
        """)
        st.dataframe(X_test)
        st.markdown("""
            **Целевая переменная (y_test)**.
            Целевая переменная представляет собой значения, которые модель пытается предсказать. 
        """)
        st.dataframe(y_test.T)
        st.header("Визуализация числовых признаков")
        selected_feature = st.selectbox(
            "Выберите признак",
            numerical_features,
            key="create_histogram_selectbox2"
        )
        st.plotly_chart(create_plot(test_data, selected_feature), use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Моделирование                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown(r"""
        ## Выбор типа математической модели прогноза

        Множественная полиноминальная регрессия представляет собой продвинутую версию линейной регрессии, предназначенную для изучения взаимосвязей между одной зависимой и несколькими независимыми переменными. Она выражается уравнением:

        $$
        y_t = \phi_0 + a_t + \sum_{i=1}^{n}{\beta_iz_i} + \sum_{j=1}^{m}{\varepsilon_jx_j^p} + \varepsilon_t
        $$

        Где:
        - $y_t$: Зависимая переменная (предсказываемая переменная).
        - $\phi_0$: Константа (Постоянный уровень ряда значение $y_t$, когда все независимые переменные равны нулю).
        - $a_t$: Тренд времени (Отражает как изменяется $y_t$ со временем. Может быть линейным, например, $a_t=\beta t$, или нелинейным, например, $a_t=\beta_1t+\beta_2t^2$).
        - Дамми переменные $z$: (обычно 0 или 1) используются для моделирования категориальных влияний на зависимую переменную.
        - Коэффициенты $\varepsilon_1,…,\varepsilon_m$: показывают величину влияния соответствующих независимых переменных на зависимую переменную.
        - Независимые переменные $x_1,…,x_m$: предполагаемые факторы, влияющие на $y_t$.
        - Полиномиальные члены: позволяют модели учитывать не только линейные, но и более сложные, нелинейные взаимосвязи между переменными.
        - Остатки модели $\varepsilon_t$: разница между наблюдаемыми значениями зависимой переменной и значениями, предсказанными моделью.

        ### Преимущества и недостатки:
        Множественная полиноминальная регрессия обладает способностью улавливать нелинейные связи и гибкостью в моделировании различных типов данных, что делает её мощным инструментом для анализа. Однако, такая модель может страдать от переобучения, проблем мультиколлинеарности и требует повышенных вычислительных ресурсов. Важно проводить регулярную оценку остатков для проверки адекватности модели и обновлять модель, включая новые данные для повышения точности прогнозов.

        Эта модель подходит для анализа сложных взаимосвязей, когда простая линейная регрессия оказывается недостаточной, и широко применяется в эконометрике, социальных науках и других областях.
    """)

    st.markdown("""
        ## Модель полиномиальной регрессии
        Модель полиномиальной регрессии используется для анализа нелинейных зависимостей между предикторами и ценой электроэнергии.
        Ниже представлены параметры модели, оценки коэффициентов, их стандартные ошибки, t-статистики и p-значения, что помогает оценить статистическую значимость каждого коэффициента.
    """)

    @st.cache_data
    def prepare_data(X_poly, y):
        model = sm.OLS(y, X_poly).fit()
        y_pred = model.predict(X_poly)
        return model, y_pred

    poly_features = PolynomialFeatures(degree=2)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.transform(X_test)

    model, y_pred_train = prepare_data(X_poly_train, y_train)

    # Извлечение данных о параметрах модели
    summary_data = model.summary().tables[1]
    info = pd.DataFrame(summary_data.data[1:], columns=summary_data.data[0])
    feature_names = poly_features.get_feature_names_out(X_train.columns)
    info['Param'] = ['Intercept' if i == 0 else feature_names[i] for i in range(len(feature_names))]

    st.subheader('Результаты модели')
    st.text(str(model.summary())[:950])
    st.subheader('Коэффициенты модели')
    st.dataframe(info, use_container_width=True, hide_index=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Прогноз на зависимые данные                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown("""
        ## Анализ точности построения модели
        Полиномиальная регрессия позволяет более точно описать зависимости в данных, что важно для прогнозирования в условиях изменчивой погоды и колебаний спроса на электроэнергию в отопительный период.
        ## Прогноз на зависимые данные
    """)

    def calculate_metrics(y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mse = mean_squared_error(y_true, y_pred)
        r_squared = r2_score(y_true, y_pred)
        return mape, mse, r_squared

    mape, mse, r_squared = calculate_metrics(y_train.Цена, y_pred_train)

    st.markdown(f"""
        ### Результаты прогноза на зависимые данные
        - **MAPE (Средняя абсолютная процентная ошибка):** {mape:.2f}%
        - **MSE (Среднеквадратичная ошибка):** {mse:.2f}
        - **R² (Коэффициент детерминации):** {r_squared:.3f}
    """)
    st.plotly_chart(plot_forecast_real_data(df, y_train, y_pred_train), use_container_width=True)

    st.plotly_chart(plot_ape_mape(y_train.Цена, y_pred_train), use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Прогноз на независимые данные                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    y_pred_test = model.predict(X_poly_test)
    mape, mse, r_squared = calculate_metrics(y_test.Цена, y_pred_test)

    st.markdown(f"""
        ### Результаты прогноза на нeзависимые данные
        - **MAPE (Средняя абсолютная процентная ошибка):** {mape:.2f}%
        - **MSE (Среднеквадратичная ошибка):** {mse:.2f}
        - **R² (Коэффициент детерминации):** {r_squared:.3f}
    """)

    st.plotly_chart(plot_forecast_test_data(df, test_data, y_pred_test), use_container_width=True)

    st.plotly_chart(plot_ape_mape(y_test.Цена, y_pred_test), use_container_width=True)

    st.markdown("""
        ## Выводы по результатам моделирования

        - **Точность модели по историческим данным:** По результатам проверки модели на исторических данных, Средняя абсолютная процентная ошибка (MAPE) составляет **0.6378%**. Это значительно ниже установленного уровня точности в 5%, что свидетельствует о высокой адекватности модели в условиях использования исторических данных.

        - **Точность модели на актуальных данных:** При применении модели к данным за 29 февраля 2024 года, MAPE составил **2.013%**. Этот результат также гораздо ниже порога в 5%, установленного для точности прогноза, что подтверждает надежность модели в условиях реального времени.

        ### Общая оценка
        Эти результаты показывают, что модель полиномиальной регрессии успешно справляется с задачей прогнозирования цен на электроэнергию, демонстрируя высокую точность как на исторических, так и на текущих данных. Точность модели значительно превосходит установленные критерии, что делает её надежным инструментом для использования в оперативном управлении и планировании.

        ### Рекомендации
        - **Регулярное обновление данных и переоценка модели:** Для поддержания высокой точности прогнозов рекомендуется регулярное обновление входных данных и повторная калибровка модели.
        - **Использование модели для оперативного реагирования:** Учитывая высокую точность модели, её можно активно использовать для оперативного реагирования на изменения в спросе и предложении на рынке электроэнергии.
    """)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Форма ввода                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    with st.form("Прогнозирование цены"):
        st.write("### Введите параметры для прогноза")
        inputs = {}
        for feature in numerical_features:
            inputs[feature] = st.number_input(
                f'Введите значение для {feature}',
                min_value=float(df[feature].min()/2),
                max_value=float(df[feature].max()*2),
                value=float(df[feature].mean()),
                step=0.01,
                key=f'num_{feature}'
            )
        st.subheader('Категориальные признаки')
        col1, col2 = st.columns(2)
        with col1:
            inputs['ДаммиЧасы'] = st.selectbox("Выберите час", options=hours_features)
            inputs['ДаммиНомерНедели'] = st.selectbox("Номер недели", options=[str(week) for week in range(1, 53)])
        with col2:
            inputs['ДаммиПогодныеУсловия'] = option_menu(
                f'Погодные условия хорошие?',
                options=['Нет', 'Да'],
                icons=["circle", "check2-circle"],
                default_index=1,
            )

        if st.form_submit_button("Прогнозировать", type='primary', use_container_width=True):
            input_df = decode_inputs(inputs, numerical_features[1:], hours_features)
            input_df = input_df.reindex(sorted(X_test.columns, reverse=True), axis=1)
            prediction = model.predict(poly_features.transform(input_df))[0]
            st.success("Прогноз успешно выполнен!")
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=prediction,
                number={"valueformat": ".2f",  "suffix": "₽"},
                delta={'reference': y_train.iloc[len(y_train)-1].Цена, 'relative': True, "valueformat": ".2f",  "suffix": "%",},
                title={"text": "Прогнозируемая цена электроэнергии"}
            ))

            fig.update_layout(paper_bgcolor="#f0f2f6", font={'color': "darkblue", 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)

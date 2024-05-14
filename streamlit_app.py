import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction

# Конфигурация страницы Streamlit
st.set_page_config(
    page_title="Прогнозирование цен на электроэнергию",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

class Menu:
    apps = [
        {
            "func"  : home.app,
            "title" : "Главная",
            "icon"  : "house-fill"
        },
        {
            "func"  : prediction.app,
            "title" : "Прогнозирование",
            "icon"  : "lightning-fill"
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons  = [app["icon"]  for app in self.apps]

            selected = option_menu(
                "Меню",
                options=titles,
                icons=icons,
                menu_icon="cast",
                default_index=0,
            )

            st.info(
                """
                ## Прогнозирование цен на электроэнергию

                Это веб-приложение предназначено для анализа и прогнозирования цен на электроэнергию. Оно включает визуализацию, статистический анализ и применение машинного обучения для точного прогнозирования.

                Проект разработан для помощи в принятии решений в сфере энергетики, обеспечивая доступ к актуальным данным о ценообразовании.
                """
            )
        return selected


if __name__ == '__main__':
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    df = pd.read_csv(current_dir / 'data.csv')

    menu = Menu()
    selected = menu.run()

    # Добавление интерфейса для загрузки файла
    st.sidebar.header('Загрузите свой файл')
    uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif df is None:
        st.sidebar.warning("Пожалуйста, загрузите файл данных.")

    for app in menu.apps:
        if app["title"] == selected:
            app["func"](df, current_dir)
            break

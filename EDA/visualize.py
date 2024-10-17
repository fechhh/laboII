import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display


def run_eda(dataframe, numeric_cols, categorical_cols, seccion="completo"):
    """
    Ejecutar EDA

    Parameters
    ----------
    df: pd.DataFrame
        Conjunto de datos de las mascotas.
    numeric_cols: list
        Listado de variables numericas a incluir en el EDA.
    categorical_cols: list
        Listado de variables categoricas a incluir en el EDA.
    seccion: str
        Seccion a mostrar en el EDA.


    Returns
    ----------
    None
    """
    # Control seccion definida
    if seccion not in [
        "completo",
        "info basica",
        "valores perdidos",
        "distribucion variables numericas",
        "distribucion variables categoricas",
        "distribucion variable respuesta",
        "relacion variables numericas con respuesta",
        "relacion variables categoricas con respuesta",
        "matriz correlacion",
    ]:
        print("Definir seccion!")

    # Formato graficos
    sns.set(style="whitegrid")

    # Informacion basica
    if seccion in ["completo", "info basica"]:
        print("Detalle conjunto de datos:")
        display(dataframe.info())
        # TODO: incluir values count para variables categoricas?

    # Valores perdidos
    # Cuidado!
    # Podria pasar que:
    # Variables texto pueden estar vacias: '' o tener un string que indique nulo 'sin valor'
    # Variables numericas pueden tener un valor 0 o 99 para indicar valores perdidos
    if seccion in ["completo", "valores perdidos"]:
        print("\Valores perdidos:")
        display(dataframe.isnull().sum())

        # Ver valores perdidos
        plt.figure(figsize=(12, 6))
        sns.heatmap(dataframe.isnull(), cbar=False, cmap="viridis")
        plt.title("Heatmap de Valores Perdidos")
        plt.show()

    # Distribucion variables numericas
    if seccion in ["completo", "distribucion variables numericas"]:
        print("\nComportamiento Variables Numéricas:")
        display(dataframe[numeric_cols].describe(include="all"))

        if len(numeric_cols) > 0:
            # Número de variables numéricas
            num_numeric = len(numeric_cols)

            # Definir el número de filas y columnas para los subplots (ejemplo: 3 columnas)
            cols = 3
            rows = math.ceil(num_numeric / cols)

            # Crear la figura para los subplots
            fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
            axes = (
                axes.flatten()
            )  # Aplanar la matriz de ejes para indexarlos fácilmente

            # Iterar por cada variable numérica
            for i, col in enumerate(numeric_cols):
                axes[i].hist(
                    dataframe[col], bins=30, color="skyblue", edgecolor="black"
                )
                axes[i].set_title(f"Distribución de {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frecuencia")

            # Eliminar subplots vacíos si hay menos variables que subplots
            for i in range(num_numeric, len(axes)):
                fig.delaxes(axes[i])

            # Ajustar la disposición y mostrar el gráfico
            plt.tight_layout()
            plt.show()

    # Distribucion variables categoricas
    if seccion in ["completo", "distribucion variables categoricas"]:
        print("\nComportamiento Variables Categóricas:")
        for col in categorical_cols:
            display(
                dataframe[col]
                .value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(0)
                .astype(int)
                .astype(str)
                + "%"
            )
        if len(categorical_cols) > 0:

            # Número de variables categóricas
            num_categorical = len(categorical_cols)

            # Definir el número de filas y columnas para los subplots (ejemplo: 3 columnas)
            cols = 3
            rows = math.ceil(num_categorical / cols)

            # Crear la figura para los subplots
            fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
            axes = (
                axes.flatten()
            )  # Aplanar la matriz de ejes para indexarlos fácilmente

            # Iterar por cada variable categórica
            for i, col in enumerate(categorical_cols):
                sns.countplot(
                    data=dataframe,
                    x=col,
                    order=dataframe[col].value_counts().index,
                    ax=axes[i],
                )
                axes[i].set_title(f"Distribución de {col}")
                axes[i].tick_params(axis="x", rotation=45)

            # Eliminar subplots vacíos si hay menos variables que subplots
            for i in range(num_categorical, len(axes)):
                fig.delaxes(axes[i])

            # Ajustar la disposición y mostrar el gráfico
            plt.tight_layout()
            plt.show()

    # Variable respuesta
    if seccion in ["completo", "distribucion variable respuesta"]:
        if "AdoptionSpeed" in dataframe.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(
                data=dataframe,
                x="AdoptionSpeed",
                # order=dataframe["AdoptionSpeed"].value_counts().index,
            )
            plt.title("Distribution of AdoptionSpeed")
            plt.show()
            # UPDATE CON GRAFICO DE DENSIDAD
            # if "AdoptionSpeed" in dataframe.columns:
            #     # Gráfico de densidad para la variable 'AdoptionSpeed'
            #     plt.figure(figsize=(10, 6))
            #     sns.kdeplot(
            #         data=dataframe,
            #         x="AdoptionSpeed",
            #         fill=True,  # Para rellenar el área bajo la curva
            #         common_norm=False,  # Para evitar normalizar entre categorías si hubiera varias
            #         bw_adjust=0.5,  # Ajusta la suavidad de la curva, puedes experimentar con este valor
            #     )
            # plt.title("Density of AdoptionSpeed")
            # plt.show()

    if seccion in ["completo", "relacion variables categoricas con respuesta"]:
        # Visualize the relationship between 'AdoptionSpeed' and other categorical columns
        # for col in categorical_cols:
        #     if col != "AdoptionSpeed":
        #         plt.figure(figsize=(12, 6))
        #         sns.countplot(
        #             data=dataframe,
        #             x=col,
        #             hue="AdoptionSpeed",
        #             order=dataframe[col].value_counts().index,
        #             stat="probability",  # Para obtener la frecuencia relativa
        #         )
        #         plt.title(f"{col} vs AdoptionSpeed")
        #         plt.xticks(rotation=45)
        #         plt.show()

        # Número de variables categóricas (excluyendo 'AdoptionSpeed')
        categorical_cols_no_target = [
            col for col in categorical_cols if col != "AdoptionSpeed"
        ]
        num_categorical_no_target = len(categorical_cols_no_target)

        # Definir el número de filas y columnas para los subplots (ejemplo: 2 columnas)
        cols = 2
        rows = math.ceil(num_categorical_no_target / cols)

        # Crear la figura para los subplots
        fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
        axes = axes.flatten()  # Aplanar la matriz de ejes para indexarlos fácilmente

        # Iterar por cada variable categórica (excluyendo 'AdoptionSpeed')
        for i, col in enumerate(categorical_cols_no_target):
            # Calcular el tamaño de cada combinación de col y AdoptionSpeed
            col_counts = (
                dataframe.groupby([col, "AdoptionSpeed"], observed=True)
                .size()
                .reset_index(name="counts")
            )

            # Calcular la proporción de cada AdoptionSpeed dentro de cada categoría de col
            col_counts["percentage"] = col_counts.groupby(col, observed=True)[
                "counts"
            ].transform(lambda x: x / x.sum())

            # Graficar usando barplot para reflejar la proporción
            sns.barplot(
                data=col_counts,
                x=col,
                y="percentage",
                hue="AdoptionSpeed",
                order=dataframe[col].value_counts().index,
                ax=axes[i],  # Añadir al subplot correspondiente
            )

            # Personalizar el gráfico
            if col in ["Type", "MaturitySize", "FurLength", "Sterilized", "Health"]:
                axes[i].set_title(
                    f"{col} vs AdoptionSpeed (Frecuencia Relativa)",
                    fontdict={"fontsize": 20, "color": "red", "fontweight": "bold"},
                )
            else:
                axes[i].set_title(
                    f"{col} vs AdoptionSpeed (Frecuencia Relativa)",
                    # fontdict={"fontsize": 14, "color": "black"},
                )
            axes[i].set_ylabel("Frecuencia Relativa")
            axes[i].tick_params(
                axis="x", rotation=45
            )  # Ajustar las etiquetas del eje X

        # Eliminar subplots vacíos si hay menos variables que subplots
        for i in range(num_categorical_no_target, len(axes)):
            fig.delaxes(axes[i])

        # Ajustar la disposición y mostrar los gráficos
        plt.tight_layout()
        plt.show()

    # Visualize the relationship between 'AdoptionSpeed' and numeric columns
    if seccion in ["completo", "relacion variables numericas con respuesta"]:
        # Número de variables numéricas
        num_numeric = len(numeric_cols)

        # Definir el número de filas y columnas para los subplots (ejemplo: 2 columnas)
        cols = 2
        rows = math.ceil(num_numeric / cols)

        # Crear la figura para los subplots
        fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
        axes = axes.flatten()  # Aplanar la matriz de ejes para indexarlos fácilmente

        # Iterar por cada variable numérica
        for i, col in enumerate(numeric_cols):
            # Crear el boxplot
            sns.boxplot(data=dataframe, x="AdoptionSpeed", y=col, ax=axes[i])

            # Añadir título
            axes[i].set_title(f"{col} vs AdoptionSpeed")

        # Eliminar subplots vacíos si hay menos variables que subplots
        for i in range(num_numeric, len(axes)):
            fig.delaxes(axes[i])

        # Ajustar la disposición y mostrar los gráficos
        plt.tight_layout()
        plt.show()

    # Matriz correlacion para variables numericas (puede haber variables donde esto no tenga sentido)
    if seccion in ["completo", "matriz correlacion"]:
        # Ej. color
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = dataframe[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            plt.show()

    # # Pair plot for numeric columns
    # if len(numeric_cols) > 1:
    #     sns.pairplot(dataframe[numeric_cols])
    #     plt.suptitle("Pair Plot of Numeric Columns", y=1.02)
    #     plt.show()

    # print("EDA completed.")


# Example usage
# df = pd.read_csv('path_to_your_file.csv')
# run_eda(df)


# TODO: considerar para data cleaning...
def transform_original_columns(df):
    """
    Transforma las variables del DataFrame en categóricas o texto según corresponda.

    Parameters:
    df (pd.DataFrame): El DataFrame que contiene los datos de las mascotas.

    Returns:
    pd.DataFrame: El DataFrame con las columnas transformadas.
    """
    # Definir las transformaciones para cada columna categórica
    categorical_columns = {
        "Type": {1: "Dog", 2: "Cat"},
        "Gender": {1: "Male", 2: "Female", 3: "Mixed"},
        "MaturitySize": {
            1: "Small",
            2: "Medium",
            3: "Large",
            4: "Extra Large",
            0: "Not Specified",
        },
        "FurLength": {1: "Short", 2: "Medium", 3: "Long", 0: "Not Specified"},
        "Vaccinated": {1: "Yes", 2: "No", 3: "Not Sure"},
        "Dewormed": {1: "Yes", 2: "No", 3: "Not Sure"},
        "Sterilized": {1: "Yes", 2: "No", 3: "Not Sure"},
        "Health": {
            1: "Healthy",
            2: "Minor Injury",
            3: "Serious Injury",
            0: "Not Specified",
        },
        # "Fee": {0: "Free"},
    }

    # Transformar las variables categóricas usando el diccionario
    for col, mapping in categorical_columns.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("category")

    # Variables a texto # TODO: VER SI LO INCLUIMOS....
    # text_columns = ["PetID", "Name", "RescuerID", "Description"]
    # for col in text_columns:
    #    if col in df.columns:
    #        df[col] = df[col].astype(str)

    # La variable 'State' debería ser transformada en categórica según un diccionario específico de estados.
    # Ejemplo (necesita ajustar según el diccionario de estados real):
    # if 'State' in df.columns:
    #     df['State'] = df['State'].astype('category')

    return df

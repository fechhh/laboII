import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def run_eda(dataframe):
    """
    Ejecutar EDA

    Parameters:
    df (pd.DataFrame): El DataFrame que contiene los datos de las mascotas.

    Returns:
    None
    """
    # Formato graficos
    sns.set(style="whitegrid")

    # Formato variables categoricas que se encuentran como numericas
    dataframe = transform_columns(dataframe)

    # variable numericas
    numeric_cols = dataframe.select_dtypes(include="number").columns

    # variables categoricas
    categorical_cols = [
        "Type",
        "Gender",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Dewormed",
        "Sterilized",
        "Health",
        # "Fee", # es numerica
    ]  # dataframe.select_dtypes(include="object").columns

    # filtrar campos que parecen ser numericos pero no lo son
    campos_num_categ = [
        "Breed1",
        "Breed2",
        "Color1",
        "Color2",
        "Color3",
        "State",
        # "VideoAmt",
        # "PhotoAmt",
        "AdoptionSpeed",
    ]
    numeric_cols = [col for col in numeric_cols if col not in campos_num_categ]

    # Informacion basica
    print("DataFrame Information:")
    display(dataframe.info())
    print("\nDataFrame Description:")
    display(dataframe[numeric_cols].describe(include="all"))

    # Valores perdidos
    # Cuidado!
    # Podria pasar que:
    # Variables texto pueden estar vacias: '' o tener un string que indique nulo 'sin valor'
    # Variables numericas pueden tener un valor 0 o 99 para indicar valores perdidos
    print("\nMissing Values:")
    display(dataframe.isnull().sum())

    # Ver valores perdidos
    plt.figure(figsize=(12, 6))
    sns.heatmap(dataframe.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

    # Distribucion variables numericas
    if len(numeric_cols) > 0:
        dataframe[numeric_cols].hist(figsize=(12, 10), bins=30)
        plt.suptitle("Distribution of Numeric Columns")
        plt.show()

    # Distribucion variables categoricas
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(
                data=dataframe, x=col, order=dataframe[col].value_counts().index
            )
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.show()

    # Variable respuesta 'AdoptionSpeed'
    if "AdoptionSpeed" in dataframe.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=dataframe,
            x="AdoptionSpeed",
            order=dataframe["AdoptionSpeed"].value_counts().index,
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
        for col in categorical_cols:
            if col != "AdoptionSpeed":
                plt.figure(figsize=(12, 6))

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
                )

            plt.title(f"{col} vs AdoptionSpeed (Frecuencia Relativa por Categoría)")
            plt.xticks(rotation=45)
            plt.ylabel("Frecuencia Relativa")
            plt.show()

        # Visualize the relationship between 'AdoptionSpeed' and numeric columns
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=dataframe, x="AdoptionSpeed", y=col)
            plt.title(f"{col} vs AdoptionSpeed")
            plt.show()

    # Matriz correlacion para variables numericas (puede haber variables donde esto no tenga sentido)
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

    print("EDA completed.")


# Example usage
# df = pd.read_csv('path_to_your_file.csv')
# run_eda(df)


# TODO: considerar para data cleaning...
def transform_columns(df):
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

    # Variables a texto
    text_columns = ["PetID", "Name", "RescuerID", "Description"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # La variable 'State' debería ser transformada en categórica según un diccionario específico de estados.
    # Ejemplo (necesita ajustar según el diccionario de estados real):
    # if 'State' in df.columns:
    #     df['State'] = df['State'].astype('category')

    return df

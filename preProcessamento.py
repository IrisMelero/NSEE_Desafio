import pandas as pd
import numpy as np
from preparacaoDados import df_item11
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # escala variáveis numéricas para ter média 0 e desvio padrão 1
from sklearn.preprocessing import OneHotEncoder # transforma variáveis categóricas em múltiplas colunas binárias
from sklearn.compose import ColumnTransformer 


# Verificando cinformações auentes e dados nulos antes do preprocessamento
# df_item11.columns.tolist()
# print(df_item11.isna().sum())

# Separando as variáveis independentes (X) e a variável de saida (óbito)
X = df_item11.drop(columns=['OBITO'])
y = df_item11['OBITO']

# Definindo colunas numéricas e categóricas
# print(df_item11.dtypes.value_counts()) # Coletando os tipos de dados das colunas para divisão em coluna numérica e categórica
caract_categoricas = df_item11.select_dtypes(include=['object']).columns.tolist() # colunas categóricas
caract_numericas = df_item11.select_dtypes(include=['int64']).columns.tolist() # colunas numéricas 
 # convertendo colunas numéricas para float facilitando subtituição de outliers mais tarde
df_item11[caract_numericas] = df_item11[caract_numericas].astype(float) 

# Certificando remoção da coluna obito das colunas que definem as caracteristicas
if 'OBITO' in caract_numericas:
    caract_numericas.remove('OBITO')
if 'OBITO' in caract_categoricas:
    caract_categoricas.remove('OBITO')
    
# Substituição dos outliers numéricos pela média 
for col in caract_numericas: # loop de repetição
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1 # definição do intervalo interquartil
    outlier = (X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR)) # encontrar os outliers
    media = X[col].mean() # calculando medina da coluna
    X.loc[outlier, col] = media # substituição dos outliers pela media

# Transformando as colunas 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), caract_numericas),  # onehotencoder nas colunas numéricas
        ('cat', OneHotEncoder(drop='first'), caract_categoricas)  # onehotencoder nas colunas categóricas
    ])
X_preprocessed = preprocessor.fit_transform(X) # aplicando as transofrmações 

# Separando conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


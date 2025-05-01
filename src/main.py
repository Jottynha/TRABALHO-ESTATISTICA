import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# URL do dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'

# Nomes das colunas conforme a descrição do dataset
columns = ['vendor_name', 'model_name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

# Carregamento dos dados
df = pd.read_csv(url, names=columns)

# Visualização das primeiras linhas
print(df.head())

# Conversão das colunas numéricas para o tipo adequado
numeric_cols = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Verificação de valores ausentes
print(df.isnull().sum())
# Estatísticas descritivas
print(df.describe())

# Histograma das variáveis numéricas
df[numeric_cols].hist(bins=15, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Matriz de correlação
correlation_matrix = df[numeric_cols].corr()

# Heatmap da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Definição das variáveis independentes (features) e dependente (target)
X = df[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']]
y = df['ERP']

# Divisão dos dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciação e treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# Cálculo do erro quadrático médio (MSE) e do coeficiente de determinação (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erro Quadrático Médio (MSE): {mse:.2f}')
print(f'Coeficiente de Determinação (R²): {r2:.2f}')

# Gráfico de dispersão entre valores reais e previstos
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Previstos')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.show()

# Gráfico dos resíduos
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribuição dos Resíduos')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

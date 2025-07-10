import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =====================
# BANCO 1: DESEMPENHO DE CPUs
# =====================
print("=== Banco de Dados 1: Desempenho de CPUs ===")
url_cpu = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
columns_cpu = ['vendor_name', 'model_name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
df_cpu = pd.read_csv(url_cpu, names=columns_cpu)

# Conversão para numérico
numeric_cols = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
df_cpu[numeric_cols] = df_cpu[numeric_cols].apply(pd.to_numeric, errors='coerce')

print(df_cpu[numeric_cols].describe())

# Correlação para justificar a variável
print("\nCoeficientes de Correlação de Pearson (CPU):")
corr_cpu = df_cpu[numeric_cols].corr()
print(corr_cpu['ERP'])

# Escolher a variável com maior correlação com ERP 
X_cpu = df_cpu[['MMAX']]  
y_cpu = df_cpu['ERP']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cpu, y_cpu, test_size=0.2, random_state=42)

model_cpu = LinearRegression()
model_cpu.fit(Xc_train, yc_train)

yc_pred = model_cpu.predict(Xc_test)

mse_cpu = mean_squared_error(yc_test, yc_pred)
r2_cpu = r2_score(yc_test, yc_pred)

print(f'CPU (MMAX) - MSE: {mse_cpu:.2f}, R²: {r2_cpu:.2f}')
print("Intercept:", model_cpu.intercept_)
print("Coeficiente:", model_cpu.coef_)

# Gráfico de dispersão
plt.figure(figsize=(8, 6))
sns.scatterplot(x=yc_test, y=yc_pred)
plt.plot([y_cpu.min(), y_cpu.max()], [y_cpu.min(), y_cpu.max()], 'r--')
plt.xlabel('ERP Real')
plt.ylabel('ERP Previsto')
plt.title('CPU: Valores Reais vs. Previstos (MMAX)')
plt.tight_layout()
plt.show()

# Resíduos
residuals_cpu = yc_test - yc_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals_cpu, kde=True)
plt.title('CPU: Distribuição dos Resíduos')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# =====================
# BANCO 2: PREÇOS DE NOTEBOOKS
# =====================
print("\n=== Banco de Dados 2: Preço de Notebooks ===")
df_laptop = pd.read_csv('src/laptop_data.csv', encoding='latin1')

print("Colunas disponíveis:", df_laptop.columns)

df_laptop = df_laptop[['Ram', 'Weight', 'Price']].dropna()

df_laptop['Ram'] = df_laptop['Ram'].astype(str).str.replace('GB', '', regex=False).astype(int)
df_laptop['Weight'] = df_laptop['Weight'].astype(str).str.replace('kg', '', regex=False).astype(float)

print(df_laptop.describe())

print("\nCoeficientes de Correlação de Pearson (Notebook):")
corr_laptop = df_laptop.corr()
print(corr_laptop['Price'])

# Escolher a variável com maior correlação com Price
X_lap = df_laptop[['Ram']]  
y_lap = df_laptop['Price']

Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lap, y_lap, test_size=0.2, random_state=42)

model_lap = LinearRegression()
model_lap.fit(Xl_train, yl_train)

yl_pred = model_lap.predict(Xl_test)

mse_lap = mean_squared_error(yl_test, yl_pred)
r2_lap = r2_score(yl_test, yl_pred)

print(f'Notebook (Ram) - MSE: {mse_lap:.2f}, R²: {r2_lap:.2f}')
print("Intercept:", model_cpu.intercept_)
print("Coeficiente:", model_cpu.coef_)
print("Intercept:", model_lap.intercept_)
print("Coeficiente:", model_lap.coef_)

# Gráfico de dispersão
plt.figure(figsize=(8, 6))
sns.scatterplot(x=yl_test, y=yl_pred)
plt.plot([y_lap.min(), y_lap.max()], [y_lap.min(), y_lap.max()], 'r--')
plt.xlabel('Preço Real (€)')
plt.ylabel('Preço Previsto (€)')
plt.title('Notebook: Valores Reais vs. Previstos (Ram)')
plt.tight_layout()
plt.show()

# Resíduos
residuals_lap = yl_test - yl_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals_lap, kde=True)
plt.title('Notebook: Distribuição dos Resíduos')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()



  <h1>🔍 Análise de Correlação e Regressão Linear em Dados de Processadores</h1>

  <p>
    Este projeto tem como objetivo aplicar os conceitos estatísticos de <strong>correlação</strong> e <strong>regressão linear simples</strong> utilizando bibliotecas do ecossistema Python, como <code>pandas</code>, <code>seaborn</code>, <code>matplotlib</code> e <code>scikit-learn</code>. O foco está na análise de desempenho de processadores da Intel em relação ao TDP, frequência base e número de núcleos.
  </p>

  <h2>📊 Contexto</h2>
  <p>
    O trabalho foi desenvolvido como parte da disciplina de Estatística para Engenharia da Computação. A proposta é utilizar uma base de dados relevante à área, realizando uma análise quantitativa da relação entre variáveis contínuas. O dataset utilizado foi extraído do 
    <a href="https://www.kaggle.com/datasets/kristofferkirk/intel-processor-benchmark-dataset" target="_blank">Kaggle</a> e contém informações sobre processadores da Intel.
  </p>

  <h2>📁 Estrutura do Projeto</h2>
  <pre><code>
📦regressao-processadores
 ┣ 📄README.html
 ┣ 📄processador_regressao.py
 ┣ 📄intel_processors.csv
 ┗ 📄requirements.txt
  </code></pre>

  <h2>🧠 Conceitos Aplicados</h2>
  <ul>
    <li><strong>Correlação de Pearson:</strong> mede a relação linear entre duas variáveis numéricas.</li>
    <li><strong>Regressão Linear Simples:</strong> modelo que ajusta uma linha para prever uma variável dependente a partir de uma independente.</li>
    <li><strong>Visualização:</strong> gráficos de dispersão e linha de tendência.</li>
    <li><strong>Score R²:</strong> mede a explicação da variância da variável dependente.</li>
  </ul>

  <h2>📌 Requisitos</h2>
  <p>Instale as bibliotecas com:</p>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h3>requirements.txt</h3>
  <pre><code>
pandas
matplotlib
scikit-learn
seaborn
  </code></pre>

  <h2>🚀 Como Executar</h2>
  <pre><code>python processador_regressao.py</code></pre>

  <h2>📄 Exemplo de Código</h2>
  <pre><code>import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("intel_processors.csv")
X = df[['Base Frequency (GHz)']]
y = df['TDP (W)']

modelo = LinearRegression()
modelo.fit(X, y)

print("Coeficiente angular (a):", modelo.coef_[0])
print("Intercepto (b):", modelo.intercept_)
print("R²:", modelo.score(X, y))

sns.regplot(x='Base Frequency (GHz)', y='TDP (W)', data=df, ci=None, line_kws={"color":"red"})
plt.title('Regressão Linear entre Base Frequency e TDP')
plt.show()
  </code></pre>

  <h2>📈 Exemplo de Saída</h2>
  <pre><code>
Coeficiente angular (a): 21.57
Intercepto (b): -5.34
R²: 0.82
  </code></pre>

  <h2>📚 Referências</h2>
  <ul>
    <li><a href="https://scikit-learn.org/stable/" target="_blank">Scikit-learn Documentation</a></li>
    <li><a href="https://www.kaggle.com/datasets/kristofferkirk/intel-processor-benchmark-dataset" target="_blank">Kaggle Dataset</a></li>
  </ul>

  <h2>🧠 Conclusão</h2>
  <p>
    A análise demonstrou uma correlação forte entre a frequência base e o TDP dos processadores Intel. Essa relação estatística ajuda engenheiros a prever o comportamento térmico e energético baseado em parâmetros de clock, sendo essencial para otimização de sistemas.
  </p>

  <h2>👨‍💻 Autor</h2>
  <p>Projeto desenvolvido por alunos de Engenharia da Computação – CEFET-MG.</p>

</body>
</html>


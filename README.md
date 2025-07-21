# HR Analytics Challenge: Previsão de Attrition para a TechCorp Brasil

![Status](https://img.shields.io/badge/status-concluído-green)
![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)

Este projeto foi desenvolvido como trabalho final para a disciplina **Data Science Experience** da Universidade Presbiteriana Mackenzie.

---

## Acesse o Dashboard Interativo

A aplicação final, com todas as análises, simulações e previsões, está publicada e pode ser acessada publicamente no link abaixo:

**Clique aqui para acessar o dashboard https://ml-ops-pavani-dash.streamlit.app/**

---

## 1. Descrição do Projeto

A TechCorp Brasil, uma empresa de tecnologia com mais de 50.000 funcionários, enfrenta um aumento de 35% em sua taxa de rotatividade (*attrition*), gerando custos estimados em R$ 45 milhões anuais. Este projeto aborda o desafio de desenvolver uma solução de Machine Learning de ponta a ponta para prever o risco de saída de colaboradores, permitindo que o departamento de RH implemente ações de retenção proativas e data-driven.

A solução utiliza um modelo preditivo treinado com dados históricos da empresa para identificar os principais fatores que levam ao turnover e para gerar uma pontuação de risco individual para cada funcionário.

## 2. Funcionalidades do Dashboard

O dashboard interativo, construído com Streamlit, é o principal entregável do projeto e centraliza os resultados da análise, oferecendo as seguintes funcionalidades:

* **Visão Geral e KPIs:** Apresenta indicadores chave, como a taxa de attrition geral e por departamento, além de gráficos interativos sobre o perfil dos funcionários que deixaram a empresa.
* **Análise do Modelo Preditivo:** Oferece transparência sobre a lógica do modelo, exibindo as features mais importantes e um gráfico de interpretabilidade (SHAP) que explica o impacto médio de cada variável.
* **Análise de Cenários (Threshold):** Permite que o usuário de negócio ajuste o limiar de decisão do modelo e veja em tempo real o impacto na quantidade de funcionários sinalizados e nas métricas de performance (Recall vs. Precision).
* **Análise de Risco Individual:** Possibilita a seleção de um funcionário específico para visualizar sua probabilidade de saída e um gráfico de cascata (SHAP Waterfall Plot) que detalha os fatores exatos que contribuíram para sua pontuação de risco.

## 3. Estrutura do Repositório

├── analise/                # Contém os notebooks Jupyter com a análise exploratória e a modelagem.
├── dashboard/              # Contém a aplicação Streamlit (dashboard.py) e todos os artefatos necessários (.pkl, .csv).
├── .gitignore              # Arquivo para ignorar arquivos e pastas desnecessários (ex: venv).
└── README.md               # Este arquivo de documentação.

## 4. Metodologia

O projeto seguiu um pipeline completo de Data Science:

1.  **Análise Exploratória de Dados (EDA):** Investigação dos dados para identificar os principais drivers do attrition, como horas extras, cargo e estado civil.
2.  **Engenharia de Features:** Criação de 10 novas variáveis avançadas para capturar relações complexas, utilizando técnicas como **Target Encoding** e **transformações polinomiais**.
3.  **Modelagem Preditiva:** Treinamento e avaliação de 5 algoritmos de classificação, com tratamento de dados desbalanceados (usando **SMOTE**) e otimização de hiperparâmetros (com **GridSearchCV**). O modelo **XGBoost Otimizado** foi selecionado como o campeão.
4.  **Avaliação e Interpretabilidade:** Análise profunda do modelo final com **Matriz de Confusão**, análise de **Viés e Fairness**, e interpretabilidade com **SHAP** para entender as decisões do modelo.

## 5. Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Análise de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost
* **Interpretabilidade:** SHAP
* **Dashboard:** Streamlit
* **Versionamento:** Git & GitHub

## 6. Como Executar o Projeto Localmente

Para executar o dashboard no seu computador, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/rchavarria3007/ml-ops-pavani.git](https://github.com/rchavarria3007/ml-ops-pavani.git)
    cd ml-ops-pavani/dashboard
    ```
2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run dashboard.py
    ```

## 7. Autores

* Giovanna Protti (10747135)
* Ilan Schapira (10746127)
* Raul Chavarria (10742687)
* Santina Cortinove (10742029)

# %%

import pandas as pd
import numpy as np

# Tabela 1: Demografia e Contrato (RH)
dados_demo = {
    'ID_Funcionario': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Genero': ['M', 'F', 'F', 'Male', 'Fem', 'M', 'Feminino', 'M', 'F', 'M', 'Fem', 'M', 'F', 'Male', 'F', 'M', 'F', 'Masculino', 'F', 'M'],
    'Data_Admissao': [
        '2020-01-15', '2019-11-20', '2021-05-10', '2018-03-12', '2022-07-01',
        '2020-08-15', '2021-02-28', '2019-01-10', '2023-01-05', '2020-12-01',
        '2015-06-30', '2021-09-15', '2022-04-04', '2019-10-10', '2020-01-20',
        '2023-02-15', '2018-08-08', '2021-11-11', '2022-10-30', '2019-04-04'
    ],
    'Salario_Base': [
        'R$ 5000', '4200', '3.800', 'R$ 7.000', '3000', 
        '6000', 'R$ 4500', '8000', '2500', '5500', 
        '12000', '4.000', '3500', 'R$ 7500', '4800', 
        '2800', 'R$ 9000', '4100', '3200', '6200'
    ]
}

# Tabela 2: Performance e Promoção (Gestão)
dados_perf = {
    'ID_Funcionario': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 99],
    'Departamento': ['Vendas', 'RH', 'Vendas', 'TI', 'Vendas', 'TI', 'RH', 'TI', 'Vendas', 'Vendas', 'Diretoria', 'Vendas', 'RH', 'Vendas', 'Vendas', 'TI', 'Vendas', 'RH', 'TI', 'Externo'],
    'Ultima_Nota': [8.5, 9.0, 7.5, 9.5, 6.0, 8.0, 9.2, 8.8, 5.5, 7.0, 9.8, 7.2, 8.0, 8.5, 6.5, 9.0, 7.8, 9.1, 8.2, 5.0],
    'Promovido': ['Sim', 'Sim', 'Nao', 1, 0, 'Nao', 'Sim', 1, 0, 'Nao', 1, 'Nao', 'Sim', 'Nao', 0, 1, 'Nao', 'Sim', 'Nao', 0]
}

df_demo = pd.DataFrame(dados_demo)
df_perf = pd.DataFrame(dados_perf)

# %%

# 1. Limpeza Pesada (O Desafio do Salário e Gênero)

# Salário: A coluna Salario_Base é string (texto) porque tem "R$" e pontos.
# Tarefa: Remova o "R$", remova o ponto de milhar (".") e converta para float.

df_demo["Salario_Base"] = ((df_demo["Salario_Base"].str.replace("R$", "")).str.replace(".", "")).astype(float)

# %%

# Gênero: Padronize tudo para "M" e "F". (Tem "Male", "Fem", "Masculino"...).

df_demo["Genero"] = df_demo["Genero"].replace({
    "Male": "M",
    "Masculino": "M",
    "Fem": "F",
    "Feminino": "F"
})

df_demo["Genero"].value_counts()

# %%


# 2. O Merge (Juntando as pontas)

# Junte as tabelas usando ID_Funcionario.

# Regra: O RH quer analisar todos os funcionários da base demográfica. Se alguém não tiver avaliação (como o ID 14), deve aparecer com dados vazios. O ID 99 (intruso da base de performance) não deve aparecer.

df = pd.merge(df_demo, df_perf, on="ID_Funcionario", how="left")

df.info()
df.isnull().sum()
df.groupby("Departamento")["Promovido"].sum().reset_index()

# %%

# 3. Limpeza Pós-Merge (A Coluna Promovido)

# A coluna Promovido está uma bagunça: tem "Sim", "Nao", 1 e 0.
# Tarefa: Transforme tudo em 1 (Sim) e 0 (Não).

df["Promovido"] = df["Promovido"].replace({
    "Sim": 1,
    "Nao": 0
})

# %%

# 4. Feature Engineering (Tempo de Casa)

# Converta Data_Admissao para datetime.
# Crie a coluna Tempo_Meses: Diferença entre hoje (2025-01-01) e a admissão, em meses (aproximado, pode dividir os dias por 30).

df["Data_Admissao"] = pd.to_datetime(df["Data_Admissao"])

data = pd.to_datetime("2026-01-01")
df["Tempo_Meses"] = round((data - df["Data_Admissao"]).dt.days / 30, 0)

# %%

# 5. O Grande Insight (Análise)

# O Diretor quer saber: Existe diferença salarial média entre Homens e Mulheres?

df_sal_genero = df.groupby("Genero")["Salario_Base"].mean().reset_index()

import matplotlib.pyplot as plt

plt.bar(df_sal_genero["Genero"], df_sal_genero["Salario_Base"])
plt.title("Diferença salarial médio entre Homens e Mulheres")
plt.xlabel("Genero")
plt.ylabel("Média Salarial")
plt.show()

# %%

# E nas promoções? Qual gênero teve maior taxa de promoção ?

df_media_prom = df.groupby("Genero")["Promovido"].sum().reset_index()

plt.bar(df_media_prom["Genero"], df_media_prom["Promovido"])
plt.title("Taxa de Promoção entre Generos")
plt.xlabel("Genero")
plt.ylabel("Promovidos")
plt.show()

# %%

from sklearn import tree

df_arvore = df[["Salario_Base", "Ultima_Nota", "Promovido"]]

df_arvore["Promovido"] = df_arvore["Promovido"].replace({
    1: "Sim",
    0: "Nao"
})

df_arvore = df_arvore.dropna()

model = tree.DecisionTreeClassifier(max_depth=3)

features = ["Salario_Base", "Ultima_Nota"]
target = "Promovido"

X = df_arvore[features]
y = df_arvore[target]

model.fit(X, y)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True)

# %%

import pandas as pd
import numpy as np

# Tabela 1: Base de Funcionários (RH)
data_rh = {
    'ID_Agente': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
    'Nome': ['Ana', 'Beto', 'Carla', 'Daniel', 'Eva', 'Fabio', 'Gabi', 'Hugo', 'Igor', 'Julia', 'Kleber', 'Lia', 'Marcos', 'Nina', 'Otto', 'Paula', 'Quincas', 'Rita', 'Saulo', 'Tina'],
    'Data_Contratacao': ['2020-01-10', '2019-05-20', '2021-03-15', '2018-11-01', '2022-02-10', '2020-08-05', '2021-06-01', '2019-01-20', '2023-01-15', '2020-04-12', '2015-10-10', '2021-09-09', '2022-05-05', '2019-12-12', '2020-01-01', '2023-03-10', '2018-07-07', '2021-11-20', '2022-08-30', '2019-03-03'],
    'Horas_Treino_Tecnico': [10, 50, 20, 60, 5, 30, 15, 45, 2, 25, 80, 20, 10, 55, 35, 5, 70, 22, 12, 40],
    'Horas_Treino_SoftSkill': [40, 10, 35, 5, 45, 20, 40, 10, 48, 25, 5, 30, 42, 10, 20, 50, 5, 35, 38, 15],
    'Nivel_Escolaridade': ['Mestrado', 'Medio', 'Superior', 'Pos-Grad', 'Superior', 'Superior', 'Pos', 'Medio', 'Superior', 'Superior', 'Pos-Grad', 'Mestrado', 'Superior', 'Pos', 'Medio', 'Superior', 'Pos-Grad', 'Superior', 'Medio', 'Superior']
}

# Tabela 2: Base de Performance (Comercial)
data_vendas = {
    'ID_Agente': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 999], # Note o ID 117 faltando e o 999 intruso
    'Vendas_Ano_Atual': [120000, 350000, 150000, 400000, 90000, 200000, 130000, 320000, 50000, 180000, 500000, 140000, 100000, 380000, 210000, 60000, 160000, 95000, 250000, 10000],
    'Nota_Cliente': [9.5, 7.0, 9.0, 6.5, 9.8, 8.0, 9.2, 7.5, 9.9, 8.5, 6.0, 9.1, 8.8, 7.2, 8.1, 9.6, 8.9, 9.0, 8.2, 5.0]
}

df_rh = pd.DataFrame(data_rh)
df_vendas = pd.DataFrame(data_vendas)

# %%

# Merge: Junte as duas tabelas (df_rh e df_vendas).

df = pd.merge(df_rh, df_vendas, on="ID_Agente", how="left")
df.isnull().sum()
df.info()

# %%

# Escolaridade: Temos "Pos-Grad" e "Pos". Padronize tudo para "Pos-Grad".

df["Nivel_Escolaridade"] = df["Nivel_Escolaridade"].replace({
    "Pos": "Pos-Grad"	
})

df["Nivel_Escolaridade"].value_counts()

# %%

# Tempo de Casa: Converta Data_Contratacao para data. Calcule quantos anos completos a pessoa tem de casa (considere hoje como 2025-01-01).

df["Data_Contratacao"] = pd.to_datetime(df["Data_Contratacao"])

data = pd.to_datetime("2026-01-01")

df["Tempo_de_Casa_Anos"] = ((data - df["Data_Contratacao"]).dt.days / 365).astype(int)

# %%

# Perfil de Treinamento: Crie uma coluna nova chamada Foco_Treino.

# Se Horas_Treino_Tecnico > Horas_Treino_SoftSkill -> Valor: "Tecnico"
# Caso contrário -> Valor: "Comportamental"

df["Foco_Treino"] = df.apply(
    lambda x: "Tecnico"
    if (x["Horas_Treino_Tecnico"] > x["Horas_Treino_SoftSkill"])
    else "Comportamental",
    axis=1
)

# %%

# O que impacta mais as Vendas? Treino Técnico ou Soft Skill?
# Faça a correlação entre Vendas_Ano_Atual e os dois tipos de horas de treino.

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(df["Horas_Treino_Tecnico"], df["Vendas_Ano_Atual"])
plt.title("Correlação entre Vendas_Ano_Atual com Horas de Treino Tecnico")
plt.xlabel("Horas_Treino_Tecnico")
plt.ylabel("Vendas_Ano_Atual")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df["Horas_Treino_SoftSkill"], df["Vendas_Ano_Atual"])
plt.title("Correlação entre Vendas_Ano_Atual com Horas de Treino SolfSkill")
plt.xlabel("Horas_Treino_SoftSkill")
plt.ylabel("Vendas_Ano_Atual")
plt.show()

# %%

# Olhe a relação entre Nota_Cliente e Vendas_Ano_Atual. Quem vende muito tem nota boa ou ruim com o cliente?
# Use uma Árvore de Decisão para explicar isso.

plt.scatter(df["Nota_Cliente"], df["Vendas_Ano_Atual"])
plt.title("Relação entre Nota_Cliente para Vendas_Ano_Atual")
plt.xlabel("Nota_Cliente")
plt.ylabel("Vendas_Ano_Atual")
plt.show()

# %%


# ----------------------------------------------------------------

from sklearn import linear_model

df_analise = df.dropna(subset=["Vendas_Ano_Atual", "Nota_Cliente"])

X = df_analise[["Nota_Cliente"]]
y = df_analise["Vendas_Ano_Atual"]

reg = linear_model.LinearRegression()
reg.fit(X, y)

print(f"Coeficiente: {reg.coef_}")

predict_reg = reg.predict(X.drop_duplicates())

# %%

from sklearn import tree

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)


plt.plot(X["Nota_Cliente"], y, "o")
plt.grid()
plt.title("Relação Nota do Cliente Vs Vendas")
plt.xlabel("Nota_Cliente")
plt.ylabel("Vendas_Ano_Atual")

plt.plot(X.drop_duplicates()["Nota_Cliente"], predict_reg)


# %%

tree.plot_tree(arvore_full,
               feature_names=["Nota_Cliente"],
               filled=True)

# %%

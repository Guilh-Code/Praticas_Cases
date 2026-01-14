# %%

import pandas as pd

df1 = pd.read_csv("data/nexus_funcionarios.csv")
df2 = pd.read_csv("data/nexus_performance.csv")


# %%

# 1. Limpeza Nível Hard (Salário e Gênero):

# Salário: Tem "R$", tem ponto de milhar e tem vírgula decimal (ex: 5.200,50). Você precisa transformar isso em float (padrão Python usa ponto para decimal). *Dica: Remova 'R$', remova o ponto, substitua a vírgula por ponto.*

df1["Salario"] = (
    df1["Salario"]
    .str.replace("R$ ", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# %%

# Gênero: Padronize tudo para "M" e "F".

df1["Genero"] = df1["Genero"].replace({
    "Masculino": "M",
    "Male": "M",
    "Fem": "F",
    "Feminino": "F"
})

df1["Genero"].value_counts()

# %%

# 2. O Merge Estratégico:

# Junte as tabelas. Garanta que o funcionário ID 99 (que só tem avaliação mas não tem contrato) não entre na análise final (Use how='inner' ou filtre depois).

df = pd.merge(df1, df2, left_on="ID", right_on="ID_Func", how="left")


# %%

# 3. Engenharia de Features (Novas Métricas):

# Crie Custo_Por_Projeto: Salário dividido por Projetos_Entregues. (Isso mostra quem custa caro e entrega pouco).

df["Custo_Por_Projeto"] = round(df["Salario"] / df["Projetos_Entregues"], 2)

# %%

# Crie Burnout_Risk: Se Horas_Extras_Mes > 20 e Nivel_Satisfacao <= 2, marque como "Alto Risco", senão "Normal". 

df["Burnout_Risk"] = df.apply(
    lambda x: "Alto Risco"
    if (x["Horas_Extras_Mes"] > 20) & (x["Nivel_Satisfacao"] <= 2)
    else "Normal",
    axis=1
)

# %%

# 4. A Grande Análise (Groupby):

# O modelo de trabalho impacta a satisfação? Agrupe por Modelo_Trabalho e tire a média de Nivel_Satisfacao e Ultima_Avaliacao.

df_Satis =  df.groupby("Modelo_Trabalho")["Nivel_Satisfacao"].mean().reset_index()

import matplotlib.pyplot as plt

plt.bar(df_Satis["Modelo_Trabalho"], df_Satis["Nivel_Satisfacao"])
plt.title("Média de Nível de Satisfação entre Modelos de Trabalho")
plt.xlabel("Modelos de Trabalho")
plt.ylabel("Média Total")
plt.show()


df_Aval = df.groupby("Modelo_Trabalho")["Ultima_Avaliacao"].mean().reset_index()

plt.bar(df_Aval["Modelo_Trabalho"], df_Aval["Ultima_Avaliacao"])
plt.title("Média de Ultima Avaliação entre Modelos de Trabalho")
plt.xlabel("Modelos de Trabalho")
plt.ylabel("Média Total")
plt.show()

# %%

# Qual departamento faz mais hora extra?

df_dep_horas = (
        df.groupby("Departamento")["Horas_Extras_Mes"]
        .sum()
        .reset_index()
        .sort_values(by="Horas_Extras_Mes", ascending=True)
)


plt.bar(df_dep_horas["Departamento"], df_dep_horas["Horas_Extras_Mes"])
plt.title("Horas Extras de cada Departamento")
plt.xlabel("Departamentos")
plt.ylabel("Horas Extras")
plt.show()

# %%

# 5. Machine Learning (Previsão de Burnout):

# Use uma Árvore de Decisão para prever o Nivel_Satisfacao (ou o risco de burnout).

# Use variáveis como: Salario, Idade, Horas_Extras, Projetos.

# Pergunta: O que define a felicidade na Nexus Tech? É dinheiro ou é carga de trabalho?


from sklearn import tree

model = tree.DecisionTreeRegressor(max_depth=3)

features = ["Salario", "Idade", "Horas_Extras_Mes", "Projetos_Entregues"]
target = "Nivel_Satisfacao"

X = df[features]
y = df[target]

model.fit(X, y)

tree.plot_tree(model,
               feature_names=features,
               filled=True
               )

# %%

model_2 = tree.DecisionTreeClassifier()

features_2 = ["Salario", "Idade", "Horas_Extras_Mes", "Projetos_Entregues"]
target_2 = "Burnout_Risk"

X = df[features_2]
y = df[target_2]

model_2.fit(X, y)

tree.plot_tree(model_2,
               feature_names=features_2,
               class_names=model_2.classes_,
               filled=True)

# %%


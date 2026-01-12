# %%

import pandas as pd

df = pd.read_csv("data/funcionarios_full.csv")
df.info()
df.isnull().sum()

# %%

# Padronização de Categorias: A coluna Genero está uma zona ("Feminino", "F", "Fem", "Male", "M"). Padronize tudo para apenas "M" e "F".

df["Genero"] = df["Genero"].replace({
    "Feminino": "F",
    "Fem": "F",
    "Male": "M",
    "Masculino": "M"
})

df["Genero"].value_counts()

# %%

# O Bruno Lima (ID 1002) está sem Salário. O gestor disse: "Assuma que o salário dele é a média do departamento de Tecnologia".

Media_Tec = round(float(df.loc[df["Depto"] == "Tecnologia", "Salario_Mensal"].mean()), 0)
df.loc[df["Nome"] == "Bruno Lima", "Salario_Mensal"] = Media_Tec

# %%

# O Gabriel S. (ID 1007) não tem avaliação do ano anterior. Preencha com a Mediana geral para não distorcer.

Median_Avaliacao = df["Avaliacao_Ano_Anterior"].median()
df.loc[df["Nome"] == "Gabriel S.", "Avaliacao_Ano_Anterior"] = Median_Avaliacao

# %%

# A Carla Souza tem NaN em Horas Extras. Assuma que é "Nao" (padrão da empresa).

df.loc[df["Nome"] == "Carla Souza", "Horas_Extras"] = "Nao"

# %%

# Conversão de Tipos: A coluna Data_Admissao precisa virar data de verdade.

df["Data_Admissao"] = pd.to_datetime(df["Data_Admissao"])
df.info()

# %%


# ------------------------------------------------------------------ 

# Cálculo de Tempo de Casa: Crie uma coluna chamada Tempo_Casa_Anos.

Data = pd.to_datetime("2026-01-01")

df["Tempo_Casa_Anos"] = (Data - df["Data_Admissao"]).dt.days / 365


# %%

# A empresa define "Talento" como alguém que tem: Avaliacao_Ano_Anterior >= 4 E Num_Projetos >= 5.

df["Talento"] = df.apply(
    lambda x: "Sim"
    if (x["Avaliacao_Ano_Anterior"] >= 4) & (x["Num_Projetos"] >= 5)
    else "Nao",
    axis=1
)

# %%

# O Risco de Retenção

df[(df["Avaliacao_Ano_Anterior"] >= 4) & (df["Satisfacao_Trabalho"] <= 2)]

from sklearn import tree

model = tree.DecisionTreeClassifier()

features = ["Idade", "Salario_Mensal", "Avaliacao_Ano_Anterior", "Num_Projetos", "Satisfacao_Trabalho", "Tempo_Casa_Anos"]
target = "Status"

X = df[features]
y = df[target]

model.fit(X, y)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True)


# %%

# Possivel demissão/desligamento com base ao grafico
df[(df["Satisfacao_Trabalho"] <= 2.5) & (df["Num_Projetos"] <= 2.5)]

# %%

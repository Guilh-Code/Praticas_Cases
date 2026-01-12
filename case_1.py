# %%

import pandas as pd

df = pd.read_csv("data/funcionarios.csv")
df.head()
df.describe()
df.info()

# %%

# Limpeza Básica: "Guilherme, percebi que o dado do Igor Santos veio com problema no salário. Dê um jeito nisso sem excluir o funcionário da análise." (Explique qual lógica você usou).

Media_Dep = df.loc[df["Departamento"] == "Tecnologia", "Salario"].mean()
df.loc[df["Nome"] == "Igor Santos", "Salario"] = Media_Dep

# %%

# Análise por Área: "Qual departamento tem a maior média salarial?"

df.groupby("Departamento")["Salario"].mean().sort_values(ascending=False)

# %%

# Insight de Negócio: "Olhando para quem saiu da empresa (Status = 'Desligado'), você nota algum padrão rápido na 'Nota de Performance' ou no 'Tempo de Casa' comparado com quem ficou? Me diga o que os dados mostram."

from sklearn import tree

model = tree.DecisionTreeClassifier()

features = ["Departamento", "Salario", "Tempo_Casa_Meses", "Nota_Performance"]
target = "Status"

X = df[features]
y = df[target]

X["Departamento"] = X["Departamento"].replace({
    "Tecnologia": 1,
    "Comercial": 2,
    "RH": 3,
})

model.fit(X, y)

import matplotlib.pyplot as plt

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True)


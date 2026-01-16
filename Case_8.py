# %%

import pandas as pd

df1 = pd.read_csv("data/megamart_rh.csv")
df2 = pd.read_csv("data/megamart_performance.csv")

# %%

# 1. Limpeza "Extrema" (RH)

# Salário: Padronize para float. Cuidado com os milhares (ponto) e decimais (vírgula).

df1["Salario_Base"] = (
    df1["Salario_Base"]
    .str.replace("R$ ", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# %%

# Gênero: Unifique para "M" e "F".

df1["Genero"] = df1["Genero"].replace({
    "Fem": "F",
    "Feminino": "F",
    "Male": "M",
    "Masculino": "M"
})

df1["Genero"].value_counts()

# %%

# Datas: Converta Data_Nascimento e Data_Admissao para datetime.

df1["Data_Nascimento"] = pd.to_datetime(df1["Data_Nascimento"])
df1["Data_Admissao"] = pd.to_datetime(df1["Data_Admissao"])

df1.info()

# %%

# 2. Feature Engineering (Criando Variáveis de RH)

# Idade: Calcule a idade da pessoa (use hoje como 2025-01-01).

data = pd.to_datetime("2026-01-01")

df1["Idade"] = round((data - df1["Data_Nascimento"]).dt.days / 365, 0)

# %%

# Tempo de Casa: Calcule em anos.

df1["Tempo_de_Casa"] = round((data - df1["Data_Admissao"]).dt.days / 365, 0)

# %%

# Senioridade: Crie uma regra:

#   Tempo de Casa > 10 anos: "Veterano"
#   Entre 5 e 10 anos: "Pleno"
#   Menor que 5: "Novato"

df1["Senioridade"] = df1.apply(
    lambda x: "Veterano"
    if x["Tempo_de_Casa"] > 10
    else "Pleno"
    if (x["Tempo_de_Casa"] >= 5) & (x["Tempo_de_Casa"] <= 10)
    else "Novato",
    axis=1
)

df1["Senioridade"].value_counts().reset_index()

# %%

# 3. Merge e Limpeza de Departamento
# Faça o Left Join (tabela RH manda).

df = pd.merge(df1, df2, left_on="ID_Colaborador", right_on="ID_Func", how="left")

# %% 

# Departamento: Padronize! Temos "Vendas", "Comercial", "Logistica" e "Logística" (com acento).

#   Regra: "Vendas" e "Comercial" devem virar "Comercial".
#   "Logística" (com acento) deve virar "Logistica" (sem acento).

# Vendas: Quem não é de vendas (Logística, RH, TI) vai ficar com NaN em Vendas. Preencha com 0.

df["Departamento"].value_counts()

df["Departamento"] = df["Departamento"].replace({
    "Vendas": "Comercial",
    "Logística": "Logistica",
})

df["Vendas_Trimestre"] = df["Vendas_Trimestre"].fillna(0)

df["Cargo"].value_counts().reset_index()

# %%

# 4. O Grande Embate: Distância vs. Faltas

# O VP de Vendas disse que quem mora longe falta mais.

# Crie um gráfico de dispersão (Distancia_Trabalho_KM vs. Faltas_Ano) e veja se existe correlação.
# Diga quem tem razão.

import matplotlib.pyplot as plt

plt.scatter(df["Distancia_Trabalho_KM"], df["Faltas_Ano"])
plt.title("Correlação entre Distancia_Trabalho_KM vs. Faltas_Ano")
plt.xlabel("Distancia Km (Casa até Trabalho)")
plt.ylabel("Faltas no Ano")
plt.show()

import seaborn as sns

df["Faixa_Distancia"] = pd.cut(
    df["Distancia_Trabalho_KM"],
    bins=[-1, 10, 30, 50, 70, 100],
    labels=["0-10", "11-30", "31-50", "51-70", "71-100"]
)

plt.figure(figsize=(10, 5))
sns.boxplot(
    x="Faixa_Distancia",
    y="Faltas_Ano",
    data=df
)

plt.title("Distribuição de Faltas por Faixa de Distância")
plt.xlabel("Distância Casa-Trabalho (Km)")
plt.ylabel("Faltas no Ano")
plt.show()

# %%

# 5. ROI de Treinamento (O Pulo do Gato)

# Filtre apenas o departamento "Comercial".

# Crie a métrica Eficiencia_Vendas = Vendas_Trimestre / Salario_Base.

# Analise: Quem treina mais (Horas_Treinamento) tem maior eficiência? (Use gráfico ou correlação).

filtro = df["Departamento"] == "Comercial"

df[filtro]

# %%

# 6. Machine Learning: Prever Demissão (Churn)

# Use as variáveis: Idade, Distancia_Trabalho_KM, Salario_Base, Faltas_Ano, Horas_Treinamento.

# Target: Status (Converta "Ativo" para 0 e "Desligado" para 1).

# Use uma Árvore de Decisão para descobrir: Qual o perfil exato de quem pede demissão na MegaMart?

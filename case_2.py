# %%

import pandas as pd

df = pd.read_csv("data/recrutamento.csv")
df.head()
df.info()
df.isnull().sum()

# %%

# Tratamento de Datas: "Essas datas vieram como texto. Preciso que você calcule o 'Tempo de Processo' (dias entre a Inscrição e a Oferta) apenas para quem foi Contratado."

df["Data_Inscricao"] = pd.to_datetime(df["Data_Inscricao"])
df["Data_Entrevista"] = pd.to_datetime(df["Data_Entrevista"])
df["Data_Oferta"] = pd.to_datetime(df["Data_Oferta"])

filtro = df["Status_Final"] == "Contratado"

df.loc[filtro, "Tempo_de_Processo"] = (
    df.loc[filtro, "Data_Oferta"] -
    df.loc[filtro, "Data_Inscricao"]
)

df["Tempo_de_Processo_Dias"] = df["Tempo_de_Processo"].dt.days 

df.drop(columns=["Tempo_de_Processo"], inplace=True)

df

# %%

# O Gargalo: "Na média, onde a gente perde mais tempo com os contratados? Entre a 'Inscrição e Entrevista' ou entre a 'Entrevista e Oferta'?"

df.loc[filtro, "Tempo_Inscricao_entre_Entrevista"] = (
    df.loc[filtro, "Data_Entrevista"] -
    df.loc[filtro, "Data_Inscricao"]
)

df.loc[filtro, "Tempo_Entrevista_entre_Oferta"] = (
    df.loc[filtro, "Data_Oferta"] -
    df.loc[filtro, "Data_Entrevista"]
)

df["Tempo_Inscricao_entre_Entrevista"] = df["Tempo_Inscricao_entre_Entrevista"].dt.days
df["Tempo_Entrevista_entre_Oferta"] = df["Tempo_Entrevista_entre_Oferta"].dt.days

df[filtro]

# %%

media1 = float(df["Tempo_Inscricao_entre_Entrevista"].mean())
media2 = float(df["Tempo_Entrevista_entre_Oferta"].mean()) 

print(f"Média entre Data_Inscricao até a Data_Entrevista são de {media1:.2f} dias.")
print(f"Média entre Data_Entrevista até a Data_Oferta são de {media2:.2f} dias.")

# %%




# %%

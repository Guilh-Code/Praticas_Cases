# %%

import pandas as pd

# Tabela 1: Dados do RH (Engajamento vai de 0 a 100)
dados_rh = {
    'ID_Func': [101, 102, 103, 104, 105],
    'Nome': ['Ana', 'Beto', 'Carla', 'Daniel', 'Eva'],
    'Setor': ['Vendas', 'Tech', 'Tech', 'Vendas', 'RH'],
    'Engajamento': [85, 40, 95, 30, 80]
}
df_rh = pd.DataFrame(dados_rh)

# Tabela 2: Dados do Financeiro (Quem recebeu bônus)
dados_fin = {
    'ID_Func': [101, 103, 105, 106],
    'Bonus_Recebido': [5000, 12000, 3000, 7000]
}
df_fin = pd.DataFrame(dados_fin)

# %%

# Atenção: Eu quero manter todos os funcionários do RH na análise, mesmo quem não ganhou bônus (quem não ganhou deve aparecer com Bônus = 0 ou NaN, você decide como tratar, mas quero ver eles na tabela).

df =  pd.merge(df_rh, df_fin, on="ID_Func", how="outer")

filtro = df["Bonus_Recebido"].isnull()

df.loc[filtro, "Bonus_Recebido"] = 0


# %%

# "Guilherme, existe alguma relação visível entre ganhar bônus e ter engajamento alto? Quem não ganha bônus está desengajado?"

import matplotlib.pyplot as plt

plt.scatter(df["Bonus_Recebido"], df["Engajamento"])
plt.title("Correlação entre Bônus Recebido com Engajamento")
plt.xlabel("Bônus Recebido")
plt.ylabel("Engajamento")
plt.show()

# %%

# Ação: Se você tivesse que recomendar uma ação para o Diretor baseada apenas nesses 5 funcionários, o que você diria?



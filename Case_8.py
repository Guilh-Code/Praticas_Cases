# %%

import pandas as pd

df1 = pd.read_csv("data/megamart_rh.csv")
df2 = pd.read_csv("data/megamart_performance.csv")

# %%

# 1. Limpeza "Extrema" (RH)

# Salário: Padronize para float. Cuidado com os milhares (ponto) e decimais (vírgula).

# Gênero: Unifique para "M" e "F".

# Datas: Converta Data_Nascimento e Data_Admissao para datetime.

# %%

# 2. Feature Engineering (Criando Variáveis de RH)

# Idade: Calcule a idade da pessoa (use hoje como 2025-01-01).

# Tempo de Casa: Calcule em anos.

# Senioridade: Crie uma regra:

#   Tempo de Casa > 10 anos: "Veterano"
#   Entre 5 e 10 anos: "Pleno"
#   Menor que 5: "Novato"

# %%

# 3. Merge e Limpeza de Departamento
# Faça o Left Join (tabela RH manda).

# Departamento: Padronize! Temos "Vendas", "Comercial", "Logistica" e "Logística" (com acento).

#   Regra: "Vendas" e "Comercial" devem virar "Comercial".
#   "Logística" (com acento) deve virar "Logistica" (sem acento).

# Vendas: Quem não é de vendas (Logística, RH, TI) vai ficar com NaN em Vendas. Preencha com 0.

# %%

# 4. O Grande Embate: Distância vs. Faltas

# O VP de Vendas disse que quem mora longe falta mais.

# Crie um gráfico de dispersão (Distancia_Trabalho_KM vs. Faltas_Ano) e veja se existe correlação.

# Diga quem tem razão.

# %%

# 5. ROI de Treinamento (O Pulo do Gato)

# Filtre apenas o departamento "Comercial".

# Crie a métrica Eficiencia_Vendas = Vendas_Trimestre / Salario_Base.

# Analise: Quem treina mais (Horas_Treinamento) tem maior eficiência? (Use gráfico ou correlação).

# %%

# 6. Machine Learning: Prever Demissão (Churn)

# Use as variáveis: Idade, Distancia_Trabalho_KM, Salario_Base, Faltas_Ano, Horas_Treinamento.

# Target: Status (Converta "Ativo" para 0 e "Desligado" para 1).

# Use uma Árvore de Decisão para descobrir: Qual o perfil exato de quem pede demissão na MegaMart?

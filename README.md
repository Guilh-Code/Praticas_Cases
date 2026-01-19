<div align="center">

# 📊 People Analytics & Data Science: Cases Práticos

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Concluído-success)

</div>

## 💡 Sobre o Projeto
Este repositório documenta minha jornada em um **"Estágio Simulado de People Analytics"**. 

Para fugir dos tutoriais guiados e encarar problemas reais, utilizei uma IA (Gemini) atuando como um **Gestor de Dados Fictício**. A dinâmica consistiu em:
1.  Receber datasets propositalmente "sujos" (erros de formatação, nulos, datas em texto).
2.  Receber um problema de negócio complexo para resolver.
3.  Desenvolver a solução 100% "na raça" utilizando Python, sem pedir o código pronto.

Aqui você encontrará **7 Cases de Negócio** completos, indo desde a limpeza de dados até a criação de modelos preditivos para RH.

---

## 🛠 Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib
* **Machine Learning:** Scikit-Learn (Decision Trees, Linear Regression)
* **IDE:** VS Code

---

## 📂 Detalhes dos Cases (A Jornada)

### 1️⃣ Case: Análise de Turnover e Performance
**🏢 Contexto:** Analisar uma base inicial de funcionários para entender padrões de desligamento.
* **Desafio Técnico:** Limpeza de dados nulos (imputação pela média do departamento) e análise descritiva.
* **Solução ML:** Criação de uma **Árvore de Decisão** que identificou que a `Nota de Performance <= 6.75` era o principal fator de demissão, superando o salário.

### 2️⃣ Case: Gargalos no Recrutamento (Time to Hire)
**🏢 Contexto:** O RH precisava identificar onde o processo seletivo estava travando.
* **Desafio Técnico:** Conversão complexa de strings para `datetime`, cálculo de `timedelta` (dias) e filtragem de funil.
* **Insight:** Identificamos que o maior gargalo estava na triagem (Inscrição → Entrevista), com média de 12 dias, enquanto a oferta era rápida.

### 3️⃣ Case: Bônus Financeiro vs. Engajamento
**🏢 Contexto:** Cruzar dados financeiros e de RH para validar se "dinheiro compra engajamento".
* **Desafio Técnico:** Uso de `pd.merge` (Joins) e tratamento de dados faltantes pós-merge.
* **Insight:** Gráfico de dispersão comprovou correlação positiva linear: maiores bônus estavam diretamente ligados a maior engajamento.

### 4️⃣ Case: Retenção de Talentos (TechNova)
**🏢 Contexto:** Base com muitos erros de digitação. Objetivo: identificar "Talentos" em risco de saída.
* **Desafio Técnico:** Padronização de Gênero (`replace`), limpeza de Nulos condicionais e Engenharia de Atributos (criação da flag `Talento`).
* **Solução ML:** A Árvore de Decisão revelou um padrão crítico: funcionários com **Baixa Satisfação** pedem demissão, *exceto* quando estão envolvidos em muitos projetos (Risco de Burnout).

### 5️⃣ Case: ROI de Treinamento & Qualidade de Vendas (SafeGuard)
**🏢 Contexto:** Seguradora quer saber se investe em Treino Técnico ou Soft Skills.
* **Desafio Técnico:** Correlação entre múltiplas variáveis e Regressão Linear.
* **Insight (Paradoxo):**
    * Treino Técnico aumenta vendas (Correlação Positiva).
    * Treino Soft Skill não teve impacto direto.
    * **Alerta:** Vendedores com maior volume de vendas tinham as **piores notas** de clientes (Churn Risk), identificado via Regressão Linear.

### 6️⃣ Case: Diversidade & Promoções (VarejoMix)
**🏢 Contexto:** Auditoria de equidade salarial e critérios de promoção.
* **Desafio Técnico:** Limpeza pesada de strings (Salários com "R$", pontos e vírgulas) e padronização de categorias.
* **Insight:** Detectou-se que homens tinham média salarial maior, porém mulheres tinham maior taxa de promoção. A Árvore de Decisão mostrou que a promoção era baseada puramente em `Nota > 8.65`, indicando meritocracia no topo.

### 7️⃣ Case Final: Trabalho Remoto & Burnout (Nexus Tech)
**🏢 Contexto:** A polêmica "Remoto vs Presencial" e reclamações de sobrecarga.
* **Desafio Técnico:** Dataset maior e caótico. Engenharia de features (`Custo_Por_Projeto`, `Risco_Burnout`).
* **Solução ML:**
    * Validou que o modelo **Remoto** tinha maior satisfação e performance.
    * Árvore de Decisão descobriu a regra exata do Burnout: `Horas Extras > 21` era o gatilho matemático para a insatisfação, independente do modelo de trabalho.
 
### 8️⃣ Case: Conflito RH vs Vendas (MegaMart)
**🏢 Contexto:** Disputa interna entre Diretores. Vendas alegava que "quem mora longe falta mais"; RH defendia que "falta treinamento".
* **Desafio Técnico:** Limpeza "extrema" de dados (salários em formato BRL `R$ 1.500,00` convertidos para float, datas mistas), padronização de categorias e uso de **Boxplot** para análise estatística.
* **Insight (Data Viz):** O Boxplot derrubou a hipótese da distância: não houve correlação entre km e faltas.
* **Solução ML:** O `DecisionTreeClassifier` revelou a "regra oculta" de demissão da empresa:
    * **Tolerância Zero:** Quem tem `Faltas > 9` é desligado automaticamente.
    * **Fator Protetivo:** Para quem falta pouco, o **Treinamento** é crucial. Funcionários que treinam menos de 2.5h têm tolerância muito menor a faltas e rodam mais rápido.

---


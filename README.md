# 💳 Credit Scoring com LightGBM

Projeto completo de desenvolvimento de modelo de Credit Scoring com aplicação em Streamlit para escoragem e análise de risco.

---
Autor:

Marcos Fernandes Rocha. São Paulo - SP.

Cientista de Dados - Linkedin: https://www.linkedin.com/in/marcos-rocha-ciencia-de-dados

https://github.com/user-attachments/assets/3f91d022-0b7e-44e3-93ed-a4cc1e92c594


## Objetivo do Projeto

Desenvolver um modelo preditivo capaz de estimar a probabilidade de inadimplência de clientes de cartão de crédito, utilizando:

- 15 safras temporais
- 12 meses de performance
- Separação de validação Out of Time (OOT)
- Pipeline completo de pré-processamento
- LightGBM como algoritmo principal
- Aplicação executiva em Streamlit

---

## Metodologia

O projeto foi estruturado seguindo boas práticas de modelagem de risco de crédito:

### Amostragem
- As 3 últimas safras foram separadas como validação Out of Time (OOT).
- As demais safras foram utilizadas como base de desenvolvimento.

### Pré-processamento
Pipeline automatizado contendo:

- Substituição de valores nulos
- Winsorização para tratamento de outliers
- OneHotEncoding para variáveis categóricas
- Normalização
- Transformações numéricas
- Padronização

Todo o pipeline foi salvo junto com o modelo final (`model_final.pkl`).

---

## Modelo Utilizado

Foi utilizado o algoritmo **LightGBM**, escolhido por:

- Alta performance em problemas tabulares
- Capacidade de lidar com não-linearidades
- Eficiência computacional
- Forte poder de ranqueamento

---

## 📊 Avaliação do Modelo

As métricas avaliadas incluem:

- Acurácia
- AUC
- Gini
- KS
- Curva Lift

---

## 📉 Curva Lift — Interpretação

A Curva Lift avalia o poder de ranqueamento do modelo.

Ela mede quantas vezes a taxa de inadimplência nos grupos de maior risco supera a taxa média da carteira.

### 🔹 Eixo X
Decis de risco (base ordenada da maior para menor probabilidade).

- Decil 1 → 10% clientes com maior risco
- Decil 10 → 10% clientes com menor risco

### 🔹 Eixo Y
Lift = Taxa de inadimplência no decil / Taxa média geral

Um Lift elevado nos primeiros decis indica forte concentração de risco.

Isso demonstra capacidade do modelo de segmentar corretamente clientes de maior probabilidade de inadimplência.

---

## 📈 Importância das Variáveis

A importância das variáveis foi avaliada utilizando **Gain Importance**, que mede quanto cada variável reduziu erro durante o treinamento.

Isso permite identificar os principais drivers de risco da carteira.

---

## Classificação de Risco

Devido ao desbalanceamento da base, a classificação foi feita por percentil:

- Top 5% maiores scores → Alto Risco
- 95% restantes → Baixo Risco

Essa abordagem é amplamente utilizada em Credit Scoring, pois prioriza ranqueamento ao invés de limiar fixo (0.5).

---

## 💻 Aplicação Streamlit

Foi desenvolvida uma aplicação executiva contendo:

- Upload de base CSV
- Escoragem automática
- Classificação de risco
- Indicadores executivos
- Feature Importance
- Curva Lift
- Download da base escorada

---

## 🚀 Como Executar o Projeto

### 1️⃣ Criar ambiente
```bash
conda create -n credit_env python=3.10
conda activate credit_env
pip install -r requirements.txt

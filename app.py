#passos para desenvolver o algoritimo de previsao bancaria
# 1 ter uma base de dados
# 2 preparar os dados 
# 3 treinar o algoritimo 
# 4 testar o algoritimo 
# 5 fazer previsoes 
# 6 avaliar o algoritimo


import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.title("Previsão de score de crédito")
st.subheader("Dados da tabela")
# 1 ter uma base de dados
base_dados = pd.read_csv("clientes.csv")
st.dataframe(base_dados)
#st.write("dados")
#st.subheader("dados")
#print(base_dados.info())
st.subheader("Tipos de dados")
st.write("Aqui podemos ver os tipos de dados da tabela")
st.write(base_dados.dtypes)



# Meno para ilustracoa de previsao

st.write("---")
st.sidebar.title("Previsão do modelo em %")
st.write("---")


#Preparar os dados transformar dados categoricos em numericos
#labal para proficionais
label_profissao = LabelEncoder()
base_dados["profissao"] = label_profissao.fit_transform(base_dados["profissao"])

# label para mix de credito 
label_credito = LabelEncoder()
base_dados["mix_credito"] = label_credito.fit_transform(base_dados["mix_credito"])

# laber para comportamento de pagamento
label_pagamento = LabelEncoder()
base_dados["comportamento_pagamento"] = label_pagamento.fit_transform(base_dados["comportamento_pagamento"])

st.write("Transformar dados categoricos em numericos")
st.write(base_dados.dtypes)

# 3 treinar o algoritimo 
 #Divisao dos dados
y = base_dados["score_credito"]
x = base_dados.drop(columns=["score_credito", "id_cliente"])
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

 #Criar o modelo

modelo1 = KNeighborsClassifier()
modelo2 = RandomForestClassifier()
modelo3 = MLPClassifier()

with st.sidebar:
    with st.spinner("Treinando os modelos..."):
       modelo1.fit(x_treino, y_treino)
       modelo2.fit(x_treino, y_treino)
       modelo3.fit(x_treino, y_treino)

# 4 testar o algoritimo
       previsoes1 = modelo1.predict(x_teste)
       previsoes2 = modelo2.predict(x_teste)
       previsoes3 = modelo3.predict(x_teste)

# 5 fazer previsao
 # accuracy modelo

with st.sidebar:
    st.write("Modelo KNeighbors: ",f"{(accuracy_score(y_teste, previsoes1)*100):.2f}%")
    st.write("Modelo RandomForest: ",f"{(accuracy_score(y_teste, previsoes2)*100):.2f}%")
    st.write("Modelo MLP: ",f"{(accuracy_score(y_teste, previsoes3)*100):.2f}%")
    st.write("---")



# Faser a previsao de novos dados de cliente

# Usar o melhor modelo para fazer previsão de novos clientes
# melhor modelo é o modelo_arvoredecisao
# importar os novos clientes para fazer a previsao
novos_clientes = pd.read_csv("novos_clientes.csv")
st.subheader("Novos clientes")
st.dataframe(novos_clientes)

# Preparar os dados
# profissao
novos_clientes["profissao"] = label_profissao.transform (novos_clientes["profissao"])

# mix_credito
novos_clientes["mix_credito"] = label_credito.transform (novos_clientes["mix_credito"])

# comportamento_pagamento
novos_clientes["comportamento_pagamento"] = label_pagamento.transform (novos_clientes["comportamento_pagamento"])
st.dataframe(novos_clientes.dtypes)
# Fazer previsao
lista_previsao = []
nova_previsao = modelo2.predict(novos_clientes)
lista_previsao.append(nova_previsao)

with st.sidebar:
    data = {"Modelo RandomForest":lista_previsao[0]}
    df = pd.DataFrame(data)
    st.table(df)
    st.write("---")

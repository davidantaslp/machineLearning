## ---------------------------------------Estudos e exemplos de Machine Learning--------------------------------------- ##
## Fonte : https://thecleverprogrammer.com/2021/09/27/number-of-orders-prediction-with-machine-learning/

#imports

import pandas as pd
import numpy as np
import plotly.express as px #for plotting
from sklearn.model_selection import train_test_split #for splitting the data
import lightgbm as ltb # for the model - https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
import matplotlib.pyplot as plt #for plotting

#data import

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/supplement.csv")
print(data.head()) #printa as primeiras linhas 

# data.info() #informações sobre o dataset
# data.isnull().sum() #verifica se há valores nulos
# data.describe() #descrição do dataset - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html


# #graficos comentados para não ficar poluido
# pie = data["Store_Type"].value_counts() # grafico de pizza para o tipo de loja
# store = pie.index
# orders = pie.values
# fig = px.pie(data, values=orders, names=store)
# fig.show()


# pie2 = data["Location_Type"].value_counts() # grafico de pizza para localização
# location = pie2.index
# orders = pie2.values
# fig = px.pie(data, values=orders, names=location)
# fig.show()


# pie3 = data["Discount"].value_counts() # grafico de pizza para desconto
# discount = pie3.index
# orders = pie3.values
# fig = px.pie(data, values=orders, names=discount)
# fig.show()



# pie4 = data["Holiday"].value_counts() # grafico de pizza para feriado
# holiday = pie4.index
# orders = pie4.values
# fig = px.pie(data, values=orders, names=holiday)
# fig.show()

#Agora vamos configurar para treinar o modelos de ML

data["Discount"] = data["Discount"].map({"No": 0, "Yes": 1}) # mapeia os valores para 0 e 1
data["Store_Type"] = data["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4}) # mapeia os valores para 1, 2, 3 e 4
data["Location_Type"] = data["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}) # mapeia os valores para 1, 2, 3, 4 e 5
data.dropna() # remove os valores nulos

x = np.array(data[["Store_Type", "Location_Type", "Holiday", "Discount"]])  # variaveis independentes
y = np.array(data["#Order"])                                             # variavel dependente


# #dividindo os dados em treino e teste
xtrain, xtest, ytrain, ytest = train_test_split(x, 
                                                y, test_size=0.2, 
                                                random_state=42)

# #treinando o modelo
model = ltb.LGBMRegressor() # função que cria o modelo utilizando o lightgbm 
model.fit(xtrain, ytrain) # função que treina o modelo utilizando os dados de treino

ypred = model.predict(xtest) # função que faz a predição utilizando os dados de teste
data = pd.DataFrame(data={"Predicted Orders": ypred.flatten()}) # cria um dataframe com os valores preditos que correspondem a quantidade de pedidos
print(data.head())



# #indo um pouco mais a fundo, será realizado uma analise de predição considerando o desconto
xtest_with_discount = xtest.copy()
xtest_with_discount[:, -1] = 1  # Definir a última coluna para 1 (com desconto)

xtest_without_discount = xtest.copy()
xtest_without_discount[:, -1] = 0  # Definir a última coluna para 0 (sem desconto)

# Fazer previsões
ypred_with_discount = model.predict(xtest_with_discount)
ypred_without_discount = model.predict(xtest_without_discount)

# Calcular médias das previsões
avg_pred_with_discount = ypred_with_discount.mean()
avg_pred_without_discount = ypred_without_discount.mean()

# Plotar os resultados
plt.bar(['Com Desconto', 'Sem Desconto'], [avg_pred_with_discount, avg_pred_without_discount])
plt.ylabel('Número Médio de Pedidos')
plt.title('Previsão Média de Pedidos: Com e Sem Desconto')
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
"""
tensorflow.keras.models.Sequential – это базовая модель нейронной сети, 
которая, по сути, является контейнером для последовательно помещенных в нее слоев. Не имеет параметров;

tensorflow.keras.layers.Dense – это объект полносвязного слоя нейронной сети. 
Определяется следующими параметрами:

units – это количество нейронов в данном слое;

input_dim – размерность входного слоя (только для слоя, непосредственно следующего за входными данными,
для последующих слоев входная размерность определяется автоматически 
исходя из размерности предыдущего слоя);

activation – функция активации, которая будет использована для данного слоя («relu», «softmax» и др.)."""


model.add(Dense(units=800, input_dim=784, activation="relu"))
model.add(Dense(5, activation="softmax"))

import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml("mnist_784")
data = mnist.data
target = mnist.target

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

rbm = BernoulliRBM(n_components=100, n_iter=10, learning_rate=0.01, verbose=1)
rbm.fit(x_train)

train_features = rbm.transform(x_train)
test_features = rbm.transform(x_test)

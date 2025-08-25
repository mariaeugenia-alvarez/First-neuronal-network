from sklearn.datasets import fetch_openml
import numpy as np

# descargo datos y se asigna a la variable mnist
mnist = fetch_openml("mnist_784", version=1)

# extraemos datos y etiquetas de datos y los asignamos a X,y
X, y = mnist["data"].values.astype(np.float32), mnist["target"].values.astype(int)


import torch
from torch.nn import Sequential as S
from torch.nn import Linear as L
from torch.nn import ReLU as R

# defino el modelo y lo asigno a model
model = S(L(784, 128), R(), L(128, 10))

from tqdm import tqdm

# separo datos en entrenamiento (los primeros 60000) y test (los últimos 10000)
X_train, X_test = torch.from_numpy(X[:60000] / 255.0), torch.from_numpy(
    X[60000:] / 255.0
)
y_train, y_test = torch.from_numpy(y[:60000]), torch.from_numpy(y[60000:])

# defino el tamaño del batch y número de batches
bs = 32
num_batches = len(X_train) // bs

# instancio objetos de las clases 'CrossEntropyLoss' y 'Optim.Adam'
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# doble bucle para entrenar el modelo
for epoch in range(10):
    for b in tqdm(range(num_batches)):
        x = X_train[b * bs : (b + 1) * bs]
        y = y_train[b * bs : (b + 1) * bs]
        y_hat = model(x)  # obtengo la predicción del modelo

        # calculo función de perdida pasando las predicciones y etiquetas como parámetros
        loss = loss_fn(y_hat, y)

        # Los gradientes indican cuánto y en qué dirección deben cambiar esos pesos para mejorar el modelo.
        optimizer.zero_grad()  # reset del gradiente a cero
        loss.backward()  # calcula como cambiar los peso del modelo para que la siguiente predicción sea mejor
        optimizer.step()  # aplica los cambios calculados arriba
    print(f"Epoch {epoch+1} loss: {loss.item():.3f}")


# defino función para evaluar el modelo
def evaluate(model, X_test, y_test):
    acc = 0
    with torch.no_grad():
        for b in range(num_batches):
            x = X_test[b * bs : (b + 1) * bs]
            y = y_test[b * bs : (b + 1) * bs]
            y_hat = model(x)
            acc += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
    return acc


acc = evaluate(model, X_test, y_test)
print(f"Accuracy: {acc} / {len(X_test)}")

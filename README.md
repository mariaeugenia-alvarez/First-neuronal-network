
# Red neuronal para clasificación de dígitos (MNIST)

Este proyecto implementa una red neuronal simple en PyTorch para clasificar imágenes de dígitos escritos a mano usando el dataset MNIST.

## Librerías usadas en `Neuronal.py`

- `torch`
- `sklearn`
- `numpy`
- `tqdm`

## Descripción

El script descarga el dataset MNIST, prepara los datos, define y entrena una red neuronal, y finalmente evalúa la precisión del modelo sobre el conjunto de prueba.

## Requisitos

Instala las dependencias ejecutando:
```bash
pip install torch scikit-learn numpy tqdm
```

## Ejecución

Para entrenar y evaluar la red neuronal, ejecuta:
```bash
python Neuronal.py
```

## Salida esperada

El script mostrará la pérdida (`loss`) de cada época y la precisión final (`accuracy`) sobre el conjunto de prueba.
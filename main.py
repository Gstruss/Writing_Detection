# Importación de librerías necesarias
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import datetime
import platform
from sklearn.manifold import TSNE
import os
from sklearn.model_selection import train_test_split

# Carga de datos desde archivos CSV
train = pd.read_csv('train.csv')  # Carga el conjunto de entrenamiento
test = pd.read_csv('test.csv')    # Carga el conjunto de prueba

# Impresión de las dimensiones de los conjuntos de datos
# print('train:', train.shape)
# print('test:', test.shape)

# Separación de características y etiquetas
X = train.iloc[:, 1:785]  # Características (pixel values)
y = train.iloc[:, 0]      # Etiquetas (dígitos)
X_test = test.iloc[:, 0:784]  # Características del conjunto de prueba

# Normalización de los datos para TSNE y visualización
# X_tsn = X/255  # Normaliza los valores de píxeles entre 0 y 1
# tsne = TSNE()  # Inicializa el modelo TSNE
# tsne_res = tsne.fit_transform(X_tsn)  # Aplica TSNE a los datos normalizados

# Visualización de los datos en 2D
# plt.figure(figsize=(14, 12))
# plt.scatter(tsne_res[:,0], tsne_res[:,1], c=y, s=2)  # Crea un gráfico de dispersión
# plt.xticks([])  # Elimina marcas en el eje X
# plt.yticks([])  # Elimina marcas en el eje Y
# plt.colorbar()  # Añade una barra de color
# plt.show()  # Muestra el gráfico

# División del conjunto de entrenamiento en conjuntos de entrenamiento y validación
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1212)

# Impresión de las dimensiones de los conjuntos de entrenamiento y validación
# print('X_train:', X_train.shape)
# print('y_train:', y_train.shape)
# print('X_validation:', X_validation.shape)
# print('y_validation:', y_validation.shape)

# Reajuste de las matrices de datos para que representen imágenes de 28x28
x_train_re = X_train.to_numpy().reshape(33600, 28, 28)  # 33600 imágenes de 28x28
y_train_re = y_train.values  # Etiquetas para el conjunto de entrenamiento
x_validation_re = X_validation.to_numpy().reshape(8400, 28, 28)  # 8400 imágenes
y_validation_re = y_validation.values  # Etiquetas para el conjunto de validación
x_test_re = test.to_numpy().reshape(28000, 28, 28)  # 28000 imágenes del conjunto de prueba

# Guardar parámetros de imagen como constantes para uso posterior
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train_re.shape  # Obtiene el ancho y alto de la imagen
IMAGE_CHANNELS = 1  # Número de canales (1 para imágenes en escala de grises)
print('IMAGE_WIDTH:', IMAGE_WIDTH)  # Imprime el ancho de la imagen
print('IMAGE_HEIGHT:', IMAGE_HEIGHT)  # Imprime la altura de la imagen
print('IMAGE_CHANNELS:', IMAGE_CHANNELS)  # Imprime el número de canales

# Muestra una de las imágenes del conjunto de entrenamiento
pd.DataFrame(x_train_re[0])  # Convierte la imagen a un DataFrame para una mejor visualización (opcional)
plt.imshow(x_train_re[0], cmap=plt.cm.binary)  # Muestra la primera imagen en escala de grises
plt.show()  # Muestra la imagen

# Visualización de ejemplos de entrenamiento
numbers_to_display = 25  # Número de ejemplos a mostrar
num_cells = math.ceil(math.sqrt(numbers_to_display))  # Calcula el número de celdas para el gráfico
plt.figure(figsize=(10,10))  # Define el tamaño de la figura
for i in range(numbers_to_display):  # Itera para mostrar múltiples ejemplos
    plt.subplot(num_cells, num_cells, i+1)  # Crea un subplot para cada imagen
    plt.xticks([])  # Elimina marcas en el eje X
    plt.yticks([])  # Elimina marcas en el eje Y
    plt.grid(False)  # Elimina la cuadrícula
    plt.imshow(x_train_re[i], cmap=plt.cm.binary)  # Muestra la imagen
    plt.xlabel(y_train_re[i])  # Muestra la etiqueta correspondiente a la imagen
plt.show()  # Muestra todas las imágenes

# Informe de Análisis: Taller de Representaciones de Texto - Word Embeddings y Embedding Contextual

## Curso de Inteligencia Artificial

---

## Integrantes del Grupo

- **Johan Sebastian Naranjo**
- **Jhon Jamez Nieto Perez**
- **Juan Felipe Narvaez Amaya**
- **Juan Jose Urbano Perdomo**

---

## 1. Introducción

Este informe presenta el análisis detallado del notebook `Taller_Embeddings.ipynb`, un taller práctico sobre representaciones de texto mediante word embeddings. El documento explica cada sección del código, los conceptos fundamentales de embeddings y la implementación de algoritmos de aprendizaje automático para el procesamiento del lenguaje natural.

---

## 2. Resumen del Contenido

El taller está estructurado en las siguientes secciones:

| Sección | Contenido |
|---------|-----------|
| 1. Instalación de Dependencias | Instalación de librerías necesarias |
| 2. Conceptos Fundamentales | Comparación One-Hot vs Dense Embeddings |
| 3. Word2Vec - Entrenamiento Práctico | Entrenamiento de modelo Word2Vec |
| 4. Visualización con t-SNE | Reducción de dimensionalidad y visualización |
| 5. Clasificación de Textos | Uso de embeddings para clasificación |
| 6. Conclusiones | Resumen de aprendizajes |

---

## 3. Análisis Detallado del Código

### 3.1 Instalación de Dependencias (Celda 1)

```python
import subprocess
import sys

librerías = ['gensim', 'nltk', 'scikit-learn', 'matplotlib', 'numpy']

for lib in librerías:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])

print("Todas las librerías han sido instaladas.")
```

**Explicación:**
- **subprocess**: Módulo de Python que permite ejecutar comandos del sistema operativo desde Python. Se utiliza para instalar paquetes mediante pip.
- **sys.executable**: Retorna la ruta del ejecutable de Python actual, asegurando que pip se ejecute en el mismo entorno.
- **subprocess.check_call()**: Ejecuta el comando y espera a que termine. Lanza una excepción si el comando falla.

**Librerías utilizadas:**
- **gensim**: Biblioteca para modelado de temas y herramientas de NLP, utilizada para Word2Vec.
- **nltk**: Natural Language Toolkit, biblioteca fundamental para procesamiento de lenguaje natural.
- **scikit-learn**: Biblioteca de machine learning que proporciona herramientas para clasificación, regresión y reducción de dimensionalidad.
- **matplotlib**: Biblioteca para visualización de gráficos.
- **numpy**: Biblioteca fundamental para cálculos numéricos y operaciones con arrays.

---

### 3.2 Conceptos Fundamentales - One-Hot Encoding vs Dense Embeddings (Celda 2)

```python
import numpy as np

vocabulario = ['inteligencia', 'artificial', 'máquina', 'aprendizaje', 'datos']

one_hot = np.zeros(len(vocabulario))
one_hot[0] = 1

np.random.seed(42)
dense = np.random.randn(len(vocabulario))

print("Vector One-Hot para 'inteligencia':", one_hot)
print("Embedding denso:", dense)
```

**Explicación detallada:**

**Vector One-Hot:**
- Es una representación dispersa (sparse) donde solo un elemento es 1 y los demás son 0.
- Cada palabra tiene un vector del tamaño del vocabulario (5 en este caso).
- **Problema**: No captura relaciones semánticas. "inteligencia" y "artificial" son completamente independientes en esta representación.
- **Ventaja**: Simple de implementar.

**Embedding Denso:**
- Es una representación compacta donde cada palabra se representa como un vector de valores continuos.
- Los valores se inicializan aleatoriamente y se aprenden durante el entrenamiento.
- **Ventaja**: Capta relaciones semánticas entre palabras mediante el contexto.

**Código clave:**
- `np.zeros(len(vocabulario))`: Crea un array de ceros del tamaño del vocabulario.
- `np.random.seed(42)`: Fija una semilla para reproducibilidad de los resultados.
- `np.random.randn(len(vocabulario))`: Genera números aleatorios de una distribución normal estándar.

---

### 3.3 Word2Vec - Entrenamiento Práctico (Celda 3)

```python
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

corpus = """La inteligencia artificial es una rama de la informática.
El aprendizaje automático permite a las máquinas aprender de los datos.
Las redes neuronales imitan el funcionamiento del cerebro humano.
Los embeddings convierten palabras en vectores numéricos densos.
El procesamiento de lenguaje natural utiliza modelos de aprendizaje profundo.
"""

sentences = sent_tokenize(corpus.lower())
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

model = Word2Vec(sentences=tokenized_sentences, 
                 vector_size=50, 
                 window=5, 
                 min_count=1, 
                 sg=1)
```

**Explicación detallada:**

**Tokenización:**
- `sent_tokenize()`: Divide el texto en oraciones.
- `word_tokenize()`: Divide cada oración en palabras individuales.
- `corpus.lower()`: Convierte todo el texto a minúsculas para uniformidad.

**Parámetros de Word2Vec:**
- `vector_size=50`: Dimensionalidad de los vectores de palabras (cada palabra se representa como un vector de 50 valores).
- `window=5`: Tamaño de la ventana de contexto (considera 5 palabras antes y 5 después para aprender relaciones).
- `min_count=1`: Frecuencia mínima de una palabra para ser incluida en el vocabulario.
- `sg=1`: Usa Skip-gram en lugar de CBOW (Continuous Bag of Words). Skip-gram funciona mejor con datasets pequeños.

**Concepto de Skip-gram:**
- Skip-gram predije la palabra objetivo (centro) dado el contexto (palabras circundantes).
- Ventaja: Mejor para capturar relaciones semánticas complejas.

**Ejemplo de salida:**
```
Palabras similares a 'inteligencia':
  - numéricos            (0.4293)
  - lenguaje             (0.2892)
  - humano               (0.1822)
```

Esto indica que el modelo aprendió que "inteligencia" está semánticamente relacionada con "numéricos", "lenguaje" y "humano".

---

### 3.4 Visualización con t-SNE (Celda 4)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

palabras = ['inteligencia', 'aprendizaje', 'máquina', 'redes', 'datos']

vectores = [model.wv[p] for p in palabras if p in model.wv]
etiquetas = [p for p in palabras if p in model.wv]

X = np.array(vectores)

tsne = TSNE(n_components=2, perplexity=2, random_state=42, init='pca', learning_rate='auto')
vectors_2d = tsne.fit_transform(X)

plt.figure(figsize=(10, 7))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=150, color='royalblue', edgecolors='black')

for i, label in enumerate(etiquetas):
    plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', 
                 fontsize=12,
                 fontweight='bold')

plt.title("Visualización de Word Embeddings con t-SNE", fontsize=14)
plt.xlabel("Dimensión t-SNE 1")
plt.ylabel("Dimensión t-SNE 2")
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
```

**Explicación detallada:**

**¿Qué es t-SNE?**
- t-Distributed Stochastic Neighbor Embedding (t-SNE) es una técnica de reducción de dimensionalidad no lineal.
- Convierte vectores de alta dimensión (50 en este caso) a 2D para visualización.
- Preserva las relaciones de proximidad: palabras similares quedan cerca en el espacio 2D.

**Parámetros de t-SNE:**
- `n_components=2`: Reduce a 2 dimensiones para graficar.
- `perplexity=2`: Controla el balance entre estructura local y global. Para 5 puntos, se usa 2.
- `random_state=42`: Asegura reproducibilidad.
- `init='pca'`: Inicialización con PCA (más estable que aleatoria).
- `learning_rate='auto'`: Tasa de aprendizaje automática.

** Extracción de vectores:**
- `model.wv[p]`: Accede al vector de la palabra 'p' en el modelo Word2Vec entrenado.
- Solo incluye palabras que existen en el vocabulario del modelo.

**Visualización:**
- `plt.scatter()`: Crea el gráfico de dispersión.
- `plt.annotate()`: Añade etiquetas a cada punto.
- `textcoords="offset points"`: Posiciona el texto con desplazamiento.
- `xytext=(0,10)`: Desplaza el texto 10 puntos hacia arriba.

---

### 3.5 Clasificación de Textos con Embeddings (Celda 5)

```python
from sklearn.linear_model import LogisticRegression

def get_embedding_avg(texto, modelo):
    palabras = word_tokenize(texto.lower())
    vectores = [modelo.wv[p] for p in palabras if p in modelo.wv]
    if not vectores:
        return np.zeros(modelo.vector_size)
    return np.mean(vectores, axis=0)

textos = ["La inteligencia artificial es asombrosa",
    "El aprendizaje automático es el futuro",
    "No entiendo nada de esta tecnología",
    "Las máquinas son muy complicadas y malas"]
etiquetas = [1, 1, 0, 0]

X = np.array([get_embedding_avg(t, model) for t in textos])
y = np.array(etiquetas)

clf = LogisticRegression()
clf.fit(X, y)

precision = clf.score(X, y)
print(f"✓ Clasificador entrenado")
print(f"Precisión en entrenamiento: {precision * 100:.2f}%")

nuevo_texto = "Una película asombrosa"
vector_prueba = get_embedding_avg(nuevo_texto, model).reshape(1, -1)
prediccion = clf.predict(vector_prueba)

resultado = "Positiva" if prediccion[0] == 1 else "Negativa"
print(f"\nTexto de prueba: '{nuevo_texto}'")
print(f"Predicción: {resultado}")
```

**Explicación detallada:**

**Función get_embedding_avg():**
Esta función transforma un texto completo en un vector numérico representativo:

1. **Tokenización**: Convierte el texto en palabras individuales.
2. **Extracción de vectores**: Obtiene el vector de embedding de cada palabra del modelo Word2Vec.
3. **Promedio**: Calcula el promedio de todos los vectores de palabras para crear una representación del texto completo.
4. **Manejo de casos especiales**: Si ninguna palabra está en el vocabulario, retorna un vector de ceros.

**Ventaja del embedding promedio**:
- Representa el significado general del texto.
- Es simple y efectivo para tareas básicas de clasificación.

**Regresión Logística**:
- Clasificador lineal que aprende una frontera de decisión.
- `clf.fit(X, y)`: Entrena el modelo con los embeddings y etiquetas.
- `clf.score()`: Calcula la precisión en los datos de entrenamiento.
- `clf.predict()`: Predice la clase de nuevos textos.

**Ejemplo de predicción**:
- El texto "Una película asombrosa" se convierte a embedding.
- El clasificador predice si es positivo (1) o negativo (0).
- Esta es una aplicación práctica de embeddings en análisis de sentimientos.

---

## 4. Conceptos Clave Explicados

### 4.1 ¿Qué son los Word Embeddings?

Los word embeddings son representaciones vectoriales densas de palabras que capturan información semántica. A diferencia de la representación One-Hot (que tiene un 1 y muchos 0s), los embeddings usan vectores densos con valores continuos que el modelo aprende durante el entrenamiento.

### 4.2 Diferencia entre Word2Vec Skip-gram y CBOW

| Característica | Skip-gram | CBOW |
|---------------|-----------|------|
| Mecanismo | Predice contexto dado centro | Predice centro dado contexto |
| Datos necesarios | Más datos | Menos datos |
| Rendimiento con datasets pequeños | Mejor | Peor |
| Velocidad de entrenamiento | Más lento | Más rápido |

### 4.3 ¿Por qué usar t-SNE?

Los vectores de embeddings tienen alta dimensionalidad (típicamente 50-300). t-SNE permite:
- Reducir a 2D para visualización.
- Preservar relaciones de vecindad (palabras similares quedan cerca).
- Identificar clustering de conceptos relacionados.

---

## 5. Conclusiones del Análisis

El notebook implementa un flujo completo de trabajo con embeddings:

1. **Instalación robusta**: Usa subprocess para gestión de dependencias.
2. **Comparación conceptual**: Muestra claramente la diferencia entre representaciones sparse y densas.
3. **Implementación práctica**: Word2Vec con parámetros bien elegidos.
4. **Visualización**: t-SNE para interpretar resultados del modelo.
5. **Aplicación real**: Clasificación de textos usando embeddings promediados.

El código está bien estructurado y documentado, con comentarios que guían al estudiante en cada paso. Los conceptos introducidos son fundamentales para cualquier aplicación de Procesamiento del Lenguaje Natural (NLP).

---

## 6. Referencias de Librerías

| Librería | Uso en el Notebook |
|----------|-------------------|
| gensim | Word2Vec para crear embeddings |
| nltk | Tokenización de texto |
| scikit-learn | t-SNE, Regresión Logística |
| matplotlib | Visualización de gráficos |
| numpy | Operaciones con arrays y cálculos |

---

*Documento elaborado como parte del curso de Inteligencia Artificial*
*Análisis del notebook: Taller_Embeddings.ipynb*
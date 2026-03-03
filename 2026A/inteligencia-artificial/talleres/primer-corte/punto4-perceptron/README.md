# Taller: Perceptrón Simple desde cero

> **Archivo:** `perceptron.py`  
> **Librerías:** NumPy, Matplotlib  
> **Fecha:** 26 de febrero de 2026

---

## ¿Qué es un Perceptrón?

Imagina que tienes que decidir si salir a correr o no, basándote en dos cosas:

- ¿Está soleado? (sí = 1, no = 0)
- ¿Hace frío? (sí = 1, no = 0)

Un **perceptrón** hace exactamente eso: recibe varias entradas (datos), las evalúa y toma una decisión: **0 (no) o 1 (sí)**.

Es la unidad más básica de cualquier red neuronal. Aprende mirando ejemplos y corrigiendo sus errores, igual que una persona.

---

## ¿Cómo aprende? (La idea central)

El perceptrón tiene **pesos** internos, uno por cada entrada. Al principio no sabe nada (pesos = 0). Luego:

1. Mira un ejemplo y hace su predicción.
2. Compara con la respuesta correcta.
3. Si se equivocó → ajusta sus pesos un poco.
4. Repite con todos los ejemplos (eso es **una época**).
5. Sigue haciendo épocas hasta que no cometa errores, o hasta llegar al máximo (100).

```
error = respuesta_correcta - predicción_del_perceptrón

nuevo_peso = peso_actual + tasa_aprendizaje × error × entrada
```

Si no hubo error (error = 0), los pesos no cambian. Si sí hubo, se corrigen un poco.

---

## Estructura del código

```
perceptron.py
│
├── class Perceptron          ← El cerebro del programa
│   ├── __init__()            ← Configura tasa de aprendizaje y épocas
│   ├── _funcion_activacion() ← Decide si la salida es 0 o 1
│   ├── predecir()            ← Hace una predicción con los pesos actuales
│   └── entrenar()            ← Aprende mirando ejemplos y corrigiendo errores
│
├── graficar_datos_y_frontera() ← Dibuja los gráficos de resultados
│
├── dataset_AND()             ← Dataset 1: compuerta lógica AND
├── dataset_nubes()           ← Dataset 2: dos nubes de puntos aleatorios
│
├── evaluar()                 ← Muestra la tabla de predicciones y la precisión
│
└── if __name__ == "__main__" ← Punto de entrada: corre los dos experimentos
```

---

## Explicación de cada parte

### `class Perceptron`

Es la clase principal. Guarda los pesos, el sesgo y registra cuántos errores hubo en cada época.

```python
p = Perceptron(tasa_aprendizaje=0.1, epocas=100)
```

- **`tasa_aprendizaje`**: qué tan grande es el ajuste en cada corrección. Valor pequeño = aprende despacio pero con más precisión.
- **`epocas`**: máximo de vueltas que dará sobre todos los datos. Si antes de llegar a 100 ya no comete errores, **se detiene solo**.

---

### `_funcion_activacion(valor)`

```python
return 1 if valor >= 0 else 0
```

Es la "decisión final". Después de sumar todas las entradas multiplicadas por sus pesos, si el resultado es 0 o más → clase **1**. Si es negativo → clase **0**.

Se llama **función escalón** porque salta de golpe de 0 a 1.

---

### `predecir(X)`

```python
suma = np.dot(X, self.pesos) + self.sesgo
return self._funcion_activacion(suma)
```

Calcula: **entrada₁ × peso₁ + entrada₂ × peso₂ + sesgo**, y pasa ese resultado por la función escalón.

- `np.dot` es simplemente una multiplicación y suma eficiente.
- El **sesgo** (bias) es un número extra que ayuda a desplazar la línea de decisión.

---

### `entrenar(X, y)`

El corazón del algoritmo. Recorre todos los datos varias veces:

```python
for epoca in range(self.epocas):
    for muestra, etiqueta_real in zip(X, y):
        prediccion = self.predecir(muestra)
        error = etiqueta_real - prediccion
        self.pesos += self.tasa_aprendizaje * error * muestra
        self.sesgo += self.tasa_aprendizaje * error
```

Si `error = 0` → no cambia nada.  
Si `error = 1` → el perceptrón dijo 0 cuando era 1, sube los pesos.  
Si `error = -1` → el perceptrón dijo 1 cuando era 0, baja los pesos.

Al final de cada época, si los errores fueron 0, **para inmediatamente** con `break`.

---

### `graficar_datos_y_frontera()`

Genera dos gráficos en una sola ventana:

| Gráfico izquierdo | Gráfico derecho |
|---|---|
| Puntos por clase (rojo = 0, azul = 1) | Curva de errores por época |
| Línea verde = frontera de decisión aprendida | Muestra cómo bajan los errores |

La **frontera de decisión** es la línea donde el perceptrón está "en duda" (salida = 0). Todo lo que cae a un lado es clase 0, al otro lado clase 1.

---

## Los dos experimentos

### Dataset 1 — Compuerta AND

La **compuerta AND** es el ejemplo más clásico de problema linealmente separable.

| Entrada 1 | Entrada 2 | Resultado |
|:---------:|:---------:|:---------:|
| 0 | 0 | **0** |
| 0 | 1 | **0** |
| 1 | 0 | **0** |
| 1 | 1 | **1** |

Solo da 1 cuando **ambas entradas son 1**. Solo hay 4 puntos, así que el perceptrón converge muy rápido (en la época 4).

---

### Dataset 2 — Nubes de puntos sintéticos

Se generan 40 puntos aleatorios:

- **20 puntos** alrededor de la coordenada (1, 1) → clase 0
- **20 puntos** alrededor de la coordenada (4, 4) → clase 1

Como los grupos están bien separados en el espacio, el perceptrón converge en la época 3 con 100% de precisión.

Se usa `np.random.seed(42)` para que los números aleatorios sean siempre los mismos y el resultado sea reproducible.

---

## ¿Qué significa "linealmente separable"?

Que puedes trazar **una sola línea recta** que separe perfectamente las dos clases.

```
Clase 0 ●  ●        Si puedes dibujar esta línea  →  el perceptrón
          ──────     puede aprenderlo.
        ■  ■  Clase 1
```

El perceptrón **solo funciona** con problemas linealmente separables. Por ejemplo, la compuerta XOR **no** es linealmente separable y el perceptrón simple no puede aprenderla.

---

## Cómo ejecutarlo

```powershell
python perceptron.py
```

Al ejecutarlo verás en consola:

```
=======================================================
   PERCEPTRÓN SIMPLE — Implementación desde cero
=======================================================

[ DATASET 1: Compuerta AND ]
  ✓ Convergencia en época 4

  Predicciones (Compuerta AND):
  Entrada              Real   Predicho
  ----------------------------------------
  [0 0]                   0        0  ✓
  [0 1]                   0        0  ✓
  [1 0]                   0        0  ✓
  [1 1]                   1        1  ✓

  Precisión final: 4/4 = 100.0%
```

Y se abrirán **2 ventanas con gráficos**, una por cada dataset.

---

## Dependencias

```
numpy
matplotlib
```

Instalarlas si no las tienes:

```powershell
pip install numpy matplotlib
```

# Red Neuronal Multicapa (MLP) con Backpropagation

> **Archivo:** `mlp.py`  
> **Librerías:** NumPy, Matplotlib  
> **Fecha:** 26 de febrero de 2026

---

## ¿Por qué no alcanza con el Perceptrón?

Recuerda que el Perceptrón simple solo dibuja **una línea recta** para separar clases.

El problema **XOR** tiene sus puntos así:

```
Entrada 1 → 0     0     1     1
Entrada 2 → 0     1     0     1
Resultado → 0     1     1     0
```

Si los dibujas en un plano:

```
        │
   1 ── × ────── ·
        │
   0 ── · ────── ×
        │
        0         1

  × = clase 1  (entradas DIFERENTES)
  · = clase 0  (entradas IGUALES)
```

Los `×` están en diagonal y los `.` en la otra diagonal.
**No existe ninguna línea recta que los separe.**
Por eso el Perceptrón simple falla con XOR, y necesitamos una red más compleja.

---

## ¿Qué es una Red Neuronal Multicapa (MLP)?

Es básicamente **varios perceptrones apilados en capas**.

```
CAPA DE ENTRADA     CAPA OCULTA      CAPA DE SALIDA
                                        
   Entrada 1  ──→  Neurona 1 ──→
                   Neurona 2 ──→   Neurona final ──→  Resultado (0 o 1)
   Entrada 2  ──→  Neurona 3 ──→
                   Neurona 4 ──→
```

- La **capa oculta** transforma las entradas en una representación intermedia.
  Gracias a ella, la red puede "doblar" el espacio y separar cosas
  que no son separables con una sola línea.
- La **capa de salida** toma esa representación y da la respuesta final.

### Arquitecturas usadas en este código

| Experimento | Arquitectura | Capas ocultas | Neuronas |
|:-----------:|:------------:|:-------------:|:--------:|
| 1 | `[2 → 4 → 1]` | 1 | 4 |
| 2 | `[2 → 8 → 4 → 1]` | 2 | 8 y 4 |

---

## ¿Cómo aprende? Las dos fases

El entrenamiento se repite una y otra vez (épocas). Cada vuelta tiene dos fases:

### Fase 1 — Forward Pass (hacia adelante)

La red **hace una predicción** pasando los datos de izquierda a derecha.

```
Entrada → [capa oculta] → [capa salida] → Predicción
```

En cada neurona se calcula:

```
z = (peso_1 × entrada_1) + (peso_2 × entrada_2) + ... + sesgo
a = sigmoid(z)   ← la salida de esa neurona
```

### Fase 2 — Backpropagation (hacia atrás)

La red **compara su predicción con la respuesta correcta**, calcula cuánto se equivocó, y luego **distribuye la culpa hacia atrás** para corregir los pesos.

```
Predicción → Error → ¿quién tuvo la culpa? → corregir pesos → repetir
```

Esto usa matemática de derivadas (regla de la cadena), pero la idea es simple:
> "Si este peso contribuyó al error → ajústalo un poco en la dirección correcta"

---

## La función Sigmoid

Es la función de activación que usa cada neurona. Convierte cualquier número en un valor entre 0 y 1:

$$\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

```
     1 ┤                          ╭──────────
  0.5  ┤─────────────────────╮────
     0 ┤──────────────────────────
          -5    -2    0    2    5   ← valor de z
```

- Número muy alto → sale casi **1**
- Número muy bajo → sale casi **0**
- z = 0 → sale exactamente **0.5**

¿Por qué usarla y no la función escalón del Perceptrón?  
Porque la sigmoid es **suave y continua**, lo que permite calcular su derivada.
Sin derivada, **no hay Backpropagation**, y sin Backprop la red no puede aprender.

### Su derivada (la necesita Backprop)

$$\text{sigmoid}'(z) = a \times (1 - a)$$

Donde `a` es la salida de sigmoid. Esto significa que si ya calculaste `a = sigmoid(z)`,
no necesitas recalcular nada: la derivada sale sola de `a`.

---

## Estructura del código

```
mlp.py
│
├── class RedNeuronal
│   ├── __init__()             ← define arquitectura, pesos y sesgos
│   ├── _sigmoid()             ← función de activación
│   ├── _sigmoid_derivada()    ← necesaria para Backprop
│   ├── forward()              ← predice pasando datos hacia adelante
│   ├── backward()             ← calcula y distribuye el error
│   ├── entrenar()             ← repite forward+backward N épocas
│   └── predecir()             ← predice con los pesos ya entrenados
│
├── dataset_XOR()              ← los 4 puntos del problema XOR
├── graficar()                 ← 2 gráficos: curva pérdida + frontera
├── evaluar()                  ← tabla de predicciones y precisión
│
└── if __name__ == "__main__"  ← entrena y muestra los 2 experimentos
```

---

## Explicación de cada parte importante

### `__init__` — Crear la red

```python
red = RedNeuronal(capas=[2, 4, 1], tasa_aprendizaje=0.5, epocas=10000)
```

- `capas=[2, 4, 1]` → 2 entradas, 4 neuronas ocultas, 1 salida
- Los **pesos** se inicializan con números aleatorios pequeños (no en cero,
  porque si todos empiezan igual, todas las neuronas aprenden lo mismo → inútil)
- Los **sesgos** empiezan en 0

---

### `forward()` — Predecir

```python
activaciones = [X]                           # empezamos con la entrada

for cada capa:
    z = pesos × activacion_anterior + sesgo  # suma ponderada
    a = sigmoid(z)                           # activar
    activaciones.append(a)                   # guardar para backward
```

Guardamos las activaciones de **todas** las capas porque en Backprop necesitamos
saber qué "salió" de cada capa para calcular cuánto se equivocó.

---

### `backward()` — Corregir pesos

```python
# 1. Error en la capa de salida
delta = (prediccion - real) × sigmoid'(prediccion)

# 2. Para cada capa, de atrás hacia adelante:
    gradiente_peso = delta × activacion_anterior
    gradiente_sesgo = delta

    delta = pesos.T × delta × sigmoid'(activacion)   ← propagar hacia atrás

# 3. Actualizar
    peso  -= tasa_aprendizaje × gradiente_peso
    sesgo -= tasa_aprendizaje × gradiente_sesgo
```

La clave es `delta`: es una medida de "cuánta culpa tiene esta capa".
Se calcula en la salida y se "propaga" hacia las capas anteriores, cada vez
multiplicándose por los pesos (que representan cuánto influyó cada neurona).

---

### `entrenar()` — El bucle principal

```python
for epoca in range(10000):
    activaciones = forward(X)             # predecir
    perdida = promedio((salida - real)²)  # medir error (ECM)
    backward(activaciones, y)             # corregir pesos
```

El **Error Cuadrático Medio (ECM)** mide qué tan mal lo está haciendo la red:
- Si la predicción es exactamente la correcta → ECM = 0
- Mientras más se equivoca → ECM más alto

Con cada época el ECM debería bajar (la red aprende).

---

## Los gráficos

Al ejecutar el script aparecen **2 ventanas** (una por experimento), cada una con:

### Gráfico izquierdo — Curva de aprendizaje
Muestra cómo baja el error con las épocas. Si la red está aprendiendo bien,
la curva cae rápido al inicio y luego se aplana cerca de cero.

```
Error
  │╲
  │ ╲
  │  ╲───╮
  │       ╰─────────────────
  └──────────────────────── Épocas
```

### Gráfico derecho — Frontera de decisión
Muestra el plano dividido en zonas:
- Zona **roja** = la red predice clase 0 ahí
- Zona **azul** = la red predice clase 1 ahí
- Línea **verde** = la frontera entre ambas zonas

Los 4 puntos XOR aparecen sobre el mapa. Si la red aprendió bien,
los puntos rojos están en zona roja y los azules en zona azul.

A diferencia del Perceptrón (una sola línea recta), la MLP puede generar
**fronteras curvas o en forma de "parche"**, lo que le permite resolver XOR.

---

## Hiperparámetros configurables

Son los "ajustes" que tú defines antes de entrenar. El código los acepta como parámetros:

| Hiperparámetro | Qué hace | Valor usado |
|---|---|---|
| `capas` | Define la arquitectura completa | `[2,4,1]` o `[2,8,4,1]` |
| `tasa_aprendizaje` | Qué tan grandes son los saltos al corregir pesos | `0.5` / `0.3` |
| `epocas` | Cuántas veces recorre todos los datos entrenando | `10 000` |

**¿Cómo cambiarlos?** Simplemente edita la línea donde creas la red:

```python
# Puedes poner cualquier arquitectura y valores que quieras:
red = RedNeuronal(capas=[2, 6, 6, 1], tasa_aprendizaje=0.2, epocas=5000)
```

---

## Comparación: Perceptrón vs MLP

| | Perceptrón Simple | Red MLP |
|---|---|---|
| Capas | Solo 1 (entrada + salida) | Múltiples capas |
| Frontera de decisión | Línea recta | Curva, cualquier forma |
| ¿Resuelve XOR? | ❌ No | ✅ Sí |
| Algoritmo de aprendizaje | Regla del Perceptrón | Backpropagation |
| Función de activación | Escalón (0 o 1) | Sigmoid (suave, 0 a 1) |

---

## Cómo ejecutarlo

```powershell
python mlp.py
```

Salida esperada en consola:

```
==========================================================
   RED NEURONAL MLP — Backpropagation desde cero
==========================================================

[ EXPERIMENTO 1: Arquitectura [2 → 4 → 1] ]
  1 capa oculta | 4 neuronas | lr=0.5 | 10 000 épocas

  Época     1 | Pérdida: 0.251234
  Época  1000 | Pérdida: 0.031456
  ...
  Época 10000 | Pérdida: 0.001233

  Entrada      Real   Predicho  OK
  --------------------------------------
  [0, 0]          0        0  ✓
  [0, 1]          1        1  ✓
  [1, 0]          1        1  ✓
  [1, 1]          0        0  ✓

  Precisión: 4/4 = 100.0%
```

---

## Dependencias

```powershell
pip install numpy matplotlib
```

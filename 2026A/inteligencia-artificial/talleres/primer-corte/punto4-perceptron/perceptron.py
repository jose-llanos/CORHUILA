# =============================================================================
# PERCEPTRÓN SIMPLE - Implementación desde cero
# Librerías usadas: NumPy, Matplotlib
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. CLASE PERCEPTRÓN
# =============================================================================

class Perceptron:
    """
    Perceptrón simple con aprendizaje supervisado.

    Un perceptrón es la unidad más básica de una red neuronal.
    Recibe varias entradas, las multiplica por pesos, suma todo
    y decide si la salida es 0 o 1.
    """

    def __init__(self, tasa_aprendizaje=0.1, epocas=100):
        """
        Parámetros:
        - tasa_aprendizaje: qué tan rápido aprende (entre 0 y 1)
        - epocas: cuántas veces recorre todos los datos entrenando
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.pesos = None       # se inicializan al entrenar
        self.sesgo = None       # bias (umbral)
        self.errores_por_epoca = []  # para graficar el aprendizaje

    def _funcion_activacion(self, valor):
        """
        Función escalón: si el valor es >= 0 devuelve 1, si no devuelve 0.
        Es la 'decisión' del perceptrón.
        """
        return 1 if valor >= 0 else 0

    def predecir(self, X):
        """
        Predice la clase de una o varias muestras.
        Calcula: suma(entrada * peso) + sesgo → función escalón
        """
        suma = np.dot(X, self.pesos) + self.sesgo
        return self._funcion_activacion(suma)

    def entrenar(self, X, y):
        """
        Entrena el perceptrón con los datos X (entradas) e y (etiquetas).

        Algoritmo:
        1. Inicializar pesos en 0
        2. Para cada época:
           a. Recorrer cada muestra
           b. Predecir la salida
           c. Calcular el error = real - predicho
           d. Actualizar pesos: peso += tasa * error * entrada
           e. Actualizar sesgo: sesgo += tasa * error
        """
        n_caracteristicas = X.shape[1]

        # Inicializar pesos y sesgo en cero
        self.pesos = np.zeros(n_caracteristicas)
        self.sesgo = 0

        for epoca in range(self.epocas):
            errores_en_esta_epoca = 0

            for muestra, etiqueta_real in zip(X, y):
                # Hacer predicción con los pesos actuales
                prediccion = self.predecir(muestra)

                # Calcular error
                error = etiqueta_real - prediccion

                # Actualizar pesos y sesgo si hubo error
                self.pesos += self.tasa_aprendizaje * error * muestra
                self.sesgo += self.tasa_aprendizaje * error

                # Contar si hubo error (error != 0)
                errores_en_esta_epoca += int(error != 0)

            self.errores_por_epoca.append(errores_en_esta_epoca)

            # Si no hay errores, ya aprendió todo → parar antes
            if errores_en_esta_epoca == 0:
                print(f"  ✓ Convergencia en época {epoca + 1}")
                break


# =============================================================================
# 2. FUNCIONES DE GRAFICACIÓN
# =============================================================================

def graficar_datos_y_frontera(X, y, perceptron, titulo):
    """
    Grafica los puntos del conjunto de datos y la línea de decisión
    que aprendió el perceptrón.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')

    # --- Gráfico izquierdo: datos y frontera de decisión ---
    ax1 = axes[0]

    # Separar puntos por clase
    clase_0 = X[y == 0]
    clase_1 = X[y == 1]

    ax1.scatter(clase_0[:, 0], clase_0[:, 1],
                color='red', marker='o', s=100, label='Clase 0', zorder=3)
    ax1.scatter(clase_1[:, 0], clase_1[:, 1],
                color='blue', marker='s', s=100, label='Clase 1', zorder=3)

    # Dibujar la línea de decisión
    # La frontera es donde: w0*x0 + w1*x1 + sesgo = 0
    # Despejando x1: x1 = -(w0*x0 + sesgo) / w1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_linea = np.linspace(x_min, x_max, 200)

    if perceptron.pesos[1] != 0:
        y_linea = -(perceptron.pesos[0] * x_linea + perceptron.sesgo) / perceptron.pesos[1]
        ax1.plot(x_linea, y_linea, 'g-', linewidth=2, label='Frontera de decisión')

    ax1.set_title('Datos y frontera de decisión')
    ax1.set_xlabel('Característica 1')
    ax1.set_ylabel('Característica 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Gráfico derecho: errores por época ---
    ax2 = axes[1]
    ax2.plot(range(1, len(perceptron.errores_por_epoca) + 1),
             perceptron.errores_por_epoca,
             'o-', color='orange', linewidth=2, markersize=6)
    ax2.set_title('Errores por época (aprendizaje)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Número de errores')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. CONJUNTO DE DATOS 1 — Compuerta AND
# =============================================================================
# La compuerta AND es el ejemplo más clásico de problema linealmente separable.
# Solo da 1 cuando AMBAS entradas son 1.

def dataset_AND():
    """
    Tabla de verdad de la compuerta AND:
    0 AND 0 = 0
    0 AND 1 = 0
    1 AND 0 = 0
    1 AND 1 = 1
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])
    return X, y


# =============================================================================
# 4. CONJUNTO DE DATOS 2 — Dos nubes de puntos sintéticos
# =============================================================================
# Generamos dos grupos de puntos separados en el plano.
# Clase 0: puntos cerca de (1, 1)
# Clase 1: puntos cerca de (4, 4)

def dataset_nubes():
    """
    Genera dos nubes de puntos aleatorios (pero con semilla fija
    para que siempre salga igual).
    """
    np.random.seed(42)  # semilla para reproducibilidad

    # 20 puntos para clase 0, centrados en (1, 1)
    clase_0 = np.random.randn(20, 2) * 0.6 + np.array([1, 1])

    # 20 puntos para clase 1, centrados en (4, 4)
    clase_1 = np.random.randn(20, 2) * 0.6 + np.array([4, 4])

    X = np.vstack([clase_0, clase_1])                   # juntar ambas clases
    y = np.array([0] * 20 + [1] * 20)                   # etiquetas

    return X, y


# =============================================================================
# 5. FUNCIÓN PARA EVALUAR Y MOSTRAR RESULTADOS
# =============================================================================

def evaluar(perceptron, X, y, nombre_dataset):
    """Calcula la precisión y muestra los resultados."""
    correctos = 0
    print(f"\n  Predicciones ({nombre_dataset}):")
    print(f"  {'Entrada':<20} {'Real':>6} {'Predicho':>10}")
    print(f"  {'-'*40}")

    for muestra, etiqueta in zip(X, y):
        pred = perceptron.predecir(muestra)
        estado = '✓' if pred == etiqueta else '✗'
        print(f"  {str(muestra):<20} {etiqueta:>6} {pred:>8}  {estado}")
        if pred == etiqueta:
            correctos += 1

    precision = correctos / len(y) * 100
    print(f"\n  Precisión final: {correctos}/{len(y)} = {precision:.1f}%")
    print(f"  Pesos aprendidos: {perceptron.pesos}")
    print(f"  Sesgo aprendido:  {perceptron.sesgo:.4f}")


# =============================================================================
# 6. PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("   PERCEPTRÓN SIMPLE — Implementación desde cero")
    print("=" * 55)

    # ------------------------------------------------------------------
    # EXPERIMENTO 1: Compuerta AND
    # ------------------------------------------------------------------
    print("\n[ DATASET 1: Compuerta AND ]")

    X_and, y_and = dataset_AND()

    p1 = Perceptron(tasa_aprendizaje=0.1, epocas=100)
    p1.entrenar(X_and, y_and)

    evaluar(p1, X_and, y_and, "Compuerta AND")
    graficar_datos_y_frontera(X_and, y_and, p1, "Dataset 1 — Compuerta AND")

    # ------------------------------------------------------------------
    # EXPERIMENTO 2: Nubes de puntos sintéticos
    # ------------------------------------------------------------------
    print("\n[ DATASET 2: Nubes de puntos sintéticos ]")

    X_nubes, y_nubes = dataset_nubes()

    p2 = Perceptron(tasa_aprendizaje=0.1, epocas=100)
    p2.entrenar(X_nubes, y_nubes)

    evaluar(p2, X_nubes, y_nubes, "Nubes sintéticas")
    graficar_datos_y_frontera(X_nubes, y_nubes, p2, "Dataset 2 — Nubes de puntos sintéticos")

    print("\n" + "=" * 55)
    print("   Ejecución completada.")
    print("=" * 55)

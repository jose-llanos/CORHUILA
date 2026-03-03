# =============================================================================
# RED NEURONAL MULTICAPA (MLP) CON BACKPROPAGATION
# Implementación desde cero — solo NumPy y Matplotlib
# =============================================================================
#
# ¿Por qué necesitamos esto?
#   El Perceptrón Simple NO puede resolver el problema XOR porque
#   no es linealmente separable (no se puede separar con una sola línea).
#   Una red con capas ocultas SÍ puede aprenderlo.
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# CLASE RED NEURONAL (MLP)
# =============================================================================

class RedNeuronal:
    """
    Red Neuronal Multicapa (MLP) con Backpropagation.

    La arquitectura se define con una lista de capas.
    Ejemplo:  capas = [2, 4, 1]
              └── 2 entradas → 4 neuronas ocultas → 1 salida

    Ejemplo:  capas = [2, 8, 4, 1]
              └── 2 entradas → capa oculta 8 → capa oculta 4 → 1 salida
    """

    def __init__(self, capas, tasa_aprendizaje=0.5, epocas=10000):
        """
        capas              → lista con neuronas por capa, ej: [2, 4, 1]
        tasa_aprendizaje   → qué tan rápido ajusta los pesos (0 a 1)
        epocas             → cuántas veces recorre todos los datos
        """
        self.capas = capas
        self.lr = tasa_aprendizaje
        self.epocas = epocas
        self.perdidas = []  # guardaremos el error de cada época para graficar

        # ---------------------------------------------------------------
        # Inicializar pesos y sesgos de forma aleatoria (pequeños valores)
        # ---------------------------------------------------------------
        # Para cada par de capas consecutivas creamos una matriz de pesos.
        # Dimensión: (neuronas_capa_siguiente × neuronas_capa_anterior)
        self.pesos = []
        self.sesgos = []

        for i in range(len(capas) - 1):
            w = np.random.randn(capas[i + 1], capas[i]) * 0.5  # pesos pequeños
            b = np.zeros((capas[i + 1], 1))                     # sesgos en 0
            self.pesos.append(w)
            self.sesgos.append(b)

    # -----------------------------------------------------------------------
    # FUNCIONES DE ACTIVACIÓN
    # -----------------------------------------------------------------------

    def _sigmoid(self, z):
        """
        Función sigmoid: aplana cualquier número a un rango de 0 a 1.

            sigmoid(z) = 1 / (1 + e^(-z))

        - Número muy grande  →  cerca de 1
        - Número muy pequeño →  cerca de 0
        - z = 0              →  exactamente 0.5

        Esto le da "suavidad" a las decisiones de la red, lo que permite
        calcular cómo cambiar los pesos (derivada).
        """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivada(self, a):
        """
        Derivada de sigmoid. Se necesita en Backpropagation.

        Si ya calculamos a = sigmoid(z), entonces:
            sigmoid'(z) = a × (1 - a)

        Esta derivada dice: "¿cuánto cambia la salida si cambio el peso?"
        Es el motor matemático detrás del aprendizaje.
        """
        return a * (1 - a)

    # -----------------------------------------------------------------------
    # PASO HACIA ADELANTE (Forward Pass)
    # -----------------------------------------------------------------------

    def forward(self, X):
        """
        Recorre la red de izquierda a derecha calculando la salida en
        cada neurona de cada capa.

        En cada capa:
            z = pesos × entrada_anterior + sesgo
            a = sigmoid(z)   ← salida de esa capa

        Guardamos TODAS las activaciones porque las necesitamos
        en el paso hacia atrás (backward).

        Retorna: lista de activaciones [entrada, capa1, capa2, ..., salida]
        """
        activaciones = [X]  # la primera activación es la entrada misma

        for w, b in zip(self.pesos, self.sesgos):
            z = np.dot(w, activaciones[-1]) + b   # multiplicación matricial
            a = self._sigmoid(z)                  # activar
            activaciones.append(a)

        return activaciones  # lista con la activación de cada capa

    # -----------------------------------------------------------------------
    # PASO HACIA ATRÁS (Backpropagation)
    # -----------------------------------------------------------------------

    def backward(self, activaciones, y):
        """
        Calcula cuánto contribuyó CADA PESO al error,
        y los ajusta para que el error baje.

        Idea intuitiva:
            Si la red se equivocó, ¿quién tuvo la culpa?
            Backprop reparte la culpa desde la salida hacia las primeras capas,
            usando la regla de la cadena (derivadas en cadena).

        Pasos:
        1. Calcular el error en la capa de salida → delta
        2. Retroceder capa por capa transmitiendo ese delta
        3. Actualizar pesos y sesgos con el gradiente calculado
        """
        m = y.shape[1]  # número de muestras (para promediar el gradiente)

        # ── Paso 1: Error en la capa de salida ──────────────────────────
        #   delta = (predicción - real) × sigmoid'(predicción)
        delta = (activaciones[-1] - y) * self._sigmoid_derivada(activaciones[-1])

        # ── Paso 2: Recorrer capas DE ATRÁS HACIA ADELANTE ──────────────
        for i in reversed(range(len(self.pesos))):

            # Gradiente de los pesos = delta × activación de la capa anterior
            grad_pesos = np.dot(delta, activaciones[i].T) / m
            grad_sesgo = np.sum(delta, axis=1, keepdims=True) / m

            # Propagar el error hacia la capa anterior
            # (solo si no es la primera capa — la de entrada no tiene pesos)
            if i > 0:
                delta = np.dot(self.pesos[i].T, delta) * self._sigmoid_derivada(activaciones[i])

            # ── Paso 3: Actualizar pesos y sesgos ───────────────────────
            # Restamos el gradiente (descenso de gradiente):
            # Si el gradiente es positivo → el peso estaba muy alto → bajar
            # Si el gradiente es negativo → el peso estaba muy bajo → subir
            self.pesos[i] -= self.lr * grad_pesos
            self.sesgos[i] -= self.lr * grad_sesgo

    # -----------------------------------------------------------------------
    # ENTRENAMIENTO
    # -----------------------------------------------------------------------

    def entrenar(self, X, y):
        """
        Repite Forward + Backward durante `epocas` veces.

        Cada época:
            1. Forward  → predecir
            2. Calcular pérdida (qué tan equivocado está)
            3. Backward → ajustar pesos
        """
        # Trasponemos para que cada COLUMNA sea una muestra
        # (notación matricial estándar para redes neuronales)
        Xt = X.T  # forma: (características × muestras)
        yt = y.T  # forma: (salidas × muestras)

        for epoca in range(self.epocas):

            # ── Forward ─────────────────────────────────────────────────
            activaciones = self.forward(Xt)
            salida = activaciones[-1]

            # ── Pérdida: Error Cuadrático Medio (ECM) ───────────────────
            # ECM = promedio de (predicción - real)²
            # Mientras más bajo, mejor aprende la red.
            perdida = np.mean((salida - yt) ** 2)
            self.perdidas.append(perdida)

            # ── Backward ────────────────────────────────────────────────
            self.backward(activaciones, yt)

            # Mostrar progreso cada 1000 épocas
            if (epoca + 1) % 1000 == 0 or epoca == 0:
                print(f"  Época {epoca + 1:5d} | Pérdida: {perdida:.6f}")

    def predecir(self, X):
        """
        Realiza predicciones con los pesos ya entrenados.
        Redondea la salida sigmoid a 0 o 1 (umbral en 0.5).
        """
        activaciones = self.forward(X.T)
        salida = activaciones[-1]
        return (salida >= 0.5).astype(int).flatten()


# =============================================================================
# DATASET: Problema XOR
# =============================================================================
#
# XOR (OR exclusivo): da 1 solo cuando las entradas son DIFERENTES.
#
#   0 XOR 0 = 0   (iguales    → 0)
#   0 XOR 1 = 1   (diferentes → 1)
#   1 XOR 0 = 1   (diferentes → 1)
#   1 XOR 1 = 0   (iguales    → 0)
#
# ¿Por qué el Perceptrón simple no puede resolverlo?
# Porque NO existe una línea recta que separe los 0s de los 1s.
# Si dibujas los 4 puntos, los 1s están en diagonal y los 0s en la otra.
#
#   (0,1)=1   (1,1)=0
#       ×         ·
#       ·         ×
#   (0,0)=0   (1,0)=1
#
# La red necesita una capa oculta para crear una "curva" de separación.

def dataset_XOR():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])  # columna: una salida por muestra
    return X, y


# =============================================================================
# GRAFICACIÓN
# =============================================================================

def graficar(red, X, y, titulo):
    """
    Muestra 2 gráficos:
      Izquierdo → curva de pérdida (cómo bajó el error con las épocas)
      Derecho   → frontera de decisión (qué zonas del plano clasifica como 0 o 1)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')

    # ── Gráfico 1: Curva de pérdida ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(red.perdidas, color='royalblue', linewidth=1.5)
    ax1.set_title('Curva de aprendizaje')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Error Cuadrático Medio (ECM)')
    ax1.grid(True, alpha=0.3)

    # ── Gráfico 2: Frontera de decisión ─────────────────────────────────────
    ax2 = axes[1]

    # Crear una malla densa de puntos para ver qué predice la red en cada zona
    x0 = np.linspace(-0.5, 1.5, 300)
    x1 = np.linspace(-0.5, 1.5, 300)
    xx0, xx1 = np.meshgrid(x0, x1)
    malla = np.c_[xx0.ravel(), xx1.ravel()]
    predicciones_malla = red.predecir(malla).reshape(xx0.shape)

    # Colorear zonas (rojo=clase0, azul=clase1) y línea frontera en verde
    ax2.contourf(xx0, xx1, predicciones_malla, alpha=0.25,
                 levels=[-0.5, 0.5, 1.5], colors=['#ff9999', '#9999ff'])
    ax2.contour(xx0, xx1, predicciones_malla, levels=[0.5],
                colors='green', linewidths=2)

    # Dibujar los 4 puntos XOR
    colores = ['red' if v == 0 else 'blue' for v in y.flatten()]
    ax2.scatter(X[:, 0], X[:, 1], c=colores, s=220, zorder=5,
                edgecolors='black', linewidths=1.5)

    # Anotar cada punto con su valor XOR
    notas = ['(0,0)=0', '(0,1)=1', '(1,0)=1', '(1,1)=0']
    desp   = [(-0.18, -0.13), (-0.18, 0.07), (0.05, -0.13), (0.05, 0.07)]
    for punto, texto, d in zip(X, notas, desp):
        ax2.annotate(texto, xy=punto,
                     xytext=(punto[0] + d[0], punto[1] + d[1]), fontsize=9)

    ax2.set_title('Frontera de decisión aprendida')
    ax2.set_xlabel('Entrada 1')
    ax2.set_ylabel('Entrada 2')
    leyenda = [Patch(facecolor='#ff9999', label='Clase 0 (predicha)'),
               Patch(facecolor='#9999ff', label='Clase 1 (predicha)'),
               Patch(facecolor='green',   label='Frontera de decisión')]
    ax2.legend(handles=leyenda, loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# EVALUACIÓN
# =============================================================================

def evaluar(red, X, y):
    """Imprime tabla de predicciones y la precisión final."""
    predicciones = red.predecir(X)
    print(f"\n  {'Entrada':<12} {'Real':>6} {'Predicho':>10}  OK")
    print(f"  {'-'*38}")
    correctos = 0
    for muestra, real, pred in zip(X, y.flatten(), predicciones):
        ok = '✓' if real == pred else '✗'
        print(f"  {str(list(muestra)):<12} {real:>6} {pred:>10}  {ok}")
        correctos += (real == pred)
    print(f"\n  Precisión: {correctos}/{len(y)} = {correctos / len(y) * 100:.1f}%")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    np.random.seed(0)  # fijar semilla → resultados reproducibles

    print("=" * 58)
    print("   RED NEURONAL MLP — Backpropagation desde cero")
    print("=" * 58)

    X, y = dataset_XOR()

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENTO 1: arquitectura mínima [2 → 4 → 1]
    #   Una sola capa oculta con 4 neuronas, como pide el enunciado.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ EXPERIMENTO 1: Arquitectura [2 → 4 → 1] ]")
    print("  1 capa oculta | 4 neuronas | lr=0.5 | 10 000 épocas\n")

    red1 = RedNeuronal(capas=[2, 4, 1], tasa_aprendizaje=0.5, epocas=10000)
    red1.entrenar(X, y)
    evaluar(red1, X, y)
    graficar(red1, X, y, "Experimento 1 — MLP [2 → 4 → 1]  (XOR)")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENTO 2: arquitectura más profunda [2 → 8 → 4 → 1]
    #   Dos capas ocultas: una con 8 neuronas y otra con 4.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ EXPERIMENTO 2: Arquitectura [2 → 8 → 4 → 1] ]")
    print("  2 capas ocultas | 8 y 4 neuronas | lr=0.3 | 10 000 épocas\n")

    red2 = RedNeuronal(capas=[2, 8, 4, 1], tasa_aprendizaje=0.3, epocas=10000)
    red2.entrenar(X, y)
    evaluar(red2, X, y)
    graficar(red2, X, y, "Experimento 2 — MLP [2 → 8 → 4 → 1]  (XOR)")

    print("\n" + "=" * 58)
    print("   Ejecución completada.")
    print("=" * 58)

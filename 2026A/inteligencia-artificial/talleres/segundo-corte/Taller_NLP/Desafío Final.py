import re
import os
from collections import Counter

import nltk
import spacy

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Descargas necesarias
nltk.download('stopwords', quiet=True)

# Cargar modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")


# -----------------------------
# 1. LIMPIEZA Y PREPROCESAMIENTO
# -----------------------------
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)
    texto = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', texto)
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def preprocesar_texto(texto, idioma='spanish'):
    stop_words = set(stopwords.words(idioma))
    texto_limpio = limpiar_texto(texto)
    tokens = texto_limpio.split()
    tokens_filtrados = [t for t in tokens if t not in stop_words]
    return tokens_filtrados


# -----------------------------
# 2. LECTURA DE RESEÑAS
# -----------------------------
def leer_resenas(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        resenas = [line.strip() for line in f if line.strip()]
    return resenas


# -----------------------------
# 3. EXTRACCIÓN DE ENTIDADES
# -----------------------------
def extraer_entidades(texto):
    doc = nlp(texto)
    entidades = []

    for ent in doc.ents:
        if ent.label_ in ['PER', 'ORG', 'LOC', 'MISC']:
            entidades.append((ent.text, ent.label_))

    return entidades


# -----------------------------
# 4. ANÁLISIS DE SENTIMIENTOS
# -----------------------------
PALABRAS_POSITIVAS = {
    'excelente', 'bueno', 'genial', 'rápido', 'rapido',
    'increíble', 'increible', 'recomendado', 'amable', 'eficiente',
    'maravilloso', 'perfecto', 'satisfecho', 'feliz', 'encantó', 'encanto'
}

PALABRAS_NEGATIVAS = {
    'malo', 'terrible', 'lento', 'deficiente', 'horrible',
    'pésimo', 'pesimo', 'error', 'falló', 'fallo', 'problema',
    'caro', 'demorado', 'insatisfecho', 'dañado', 'danado', 'peor'
}


def analizar_sentimiento(texto):
    tokens = preprocesar_texto(texto)

    puntaje = 0
    for token in tokens:
        if token in PALABRAS_POSITIVAS:
            puntaje += 1
        elif token in PALABRAS_NEGATIVAS:
            puntaje -= 1

    if puntaje > 0:
        etiqueta = 'positivo'
    elif puntaje < 0:
        etiqueta = 'negativo'
    else:
        etiqueta = 'neutral'

    return etiqueta, puntaje


# -----------------------------
# 5. PALABRAS CLAVE POR SENTIMIENTO
# -----------------------------
def palabras_clave_por_sentimiento(resenas_procesadas, sentimientos, top_n=10):
    claves = {
        'positivo': [],
        'negativo': [],
        'neutral': []
    }

    for tokens, (etiqueta, _) in zip(resenas_procesadas, sentimientos):
        claves[etiqueta].extend(tokens)

    resultado = {}
    for sentimiento, palabras in claves.items():
        contador = Counter(palabras)
        resultado[sentimiento] = contador.most_common(top_n)

    return resultado


# -----------------------------
# 6. ENTIDADES MÁS MENCIONADAS
# -----------------------------
def entidades_mas_mencionadas(lista_entidades, top_n=10):
    todas = []
    for entidades in lista_entidades:
        todas.extend([entidad[0] for entidad in entidades])

    contador = Counter(todas)
    return contador.most_common(top_n)


# -----------------------------
# 7. AGRUPAR RESEÑAS SIMILARES
# -----------------------------
def agrupar_resenas_similares(resenas, umbral=0.35):
    vectorizer = TfidfVectorizer(stop_words=list(stopwords.words('spanish')))
    matriz_tfidf = vectorizer.fit_transform(resenas)

    similitud = cosine_similarity(matriz_tfidf)
    visitadas = set()
    grupos = []

    for i in range(len(resenas)):
        if i in visitadas:
            continue

        grupo = [i]
        visitadas.add(i)

        for j in range(i + 1, len(resenas)):
            if j not in visitadas and similitud[i][j] >= umbral:
                grupo.append(j)
                visitadas.add(j)

        grupos.append(grupo)

    return grupos, similitud


# -----------------------------
# 8. GENERAR REPORTE EN TXT
# -----------------------------
def generar_reporte_txt(resenas, ruta_salida="reporte_resenas.txt"):
    resenas_limpias = [limpiar_texto(r) for r in resenas]
    resenas_procesadas = [preprocesar_texto(r) for r in resenas]

    sentimientos = [analizar_sentimiento(r) for r in resenas]
    entidades = [extraer_entidades(r) for r in resenas]

    claves_sentimiento = palabras_clave_por_sentimiento(resenas_procesadas, sentimientos)
    entidades_top = entidades_mas_mencionadas(entidades)

    grupos, matriz_similitud = agrupar_resenas_similares(resenas_limpias)

    puntajes = [puntaje for _, puntaje in sentimientos]
    promedio_sentimiento = sum(puntajes) / len(puntajes) if puntajes else 0

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE ANÁLISIS DE RESEÑAS\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Sentimiento promedio:\n")
        f.write(f"   {promedio_sentimiento:.2f}\n\n")

        f.write("2. Palabras clave por sentimiento:\n")
        for sentimiento, palabras in claves_sentimiento.items():
            f.write(f"\n   {sentimiento.upper()}:\n")
            for palabra, freq in palabras:
                f.write(f"   - {palabra}: {freq}\n")

        f.write("\n3. Entidades más mencionadas:\n")
        if entidades_top:
            for entidad, freq in entidades_top:
                f.write(f"   - {entidad}: {freq}\n")
        else:
            f.write("   No se encontraron entidades relevantes.\n")

        f.write("\n4. Reseñas similares agrupadas:\n")
        for idx, grupo in enumerate(grupos, 1):
            f.write(f"\n   Grupo {idx}:\n")
            for i in grupo:
                f.write(f"   - Reseña {i+1}: {resenas[i]}\n")

        f.write("\n5. Detalle por reseña:\n")
        for i, resena in enumerate(resenas):
            etiqueta, puntaje = sentimientos[i]
            f.write(f"\n   Reseña {i+1}:\n")
            f.write(f"   Texto: {resena}\n")
            f.write(f"   Sentimiento: {etiqueta} (puntaje: {puntaje})\n")
            f.write(f"   Entidades: {entidades[i]}\n")

    print(f"Reporte guardado correctamente en: {ruta_salida}")


# -----------------------------
# 9. EJECUCIÓN PRINCIPAL
# -----------------------------
if __name__ == "__main__":
    ruta = "C:/Users/Sebastian/Desktop/Taller_NLP/Reseña.txt"
    ruta_salida = "C:/Users/Sebastian/Desktop/Taller_NLP/reporte_resultados.txt"

    if os.path.exists(ruta):
        resenas = leer_resenas(ruta)
        generar_reporte_txt(resenas, ruta_salida)
    else:
        print("No se encontró el archivo 'Reseña.txt'.")
        print("Crea un archivo de texto con una reseña por línea.")
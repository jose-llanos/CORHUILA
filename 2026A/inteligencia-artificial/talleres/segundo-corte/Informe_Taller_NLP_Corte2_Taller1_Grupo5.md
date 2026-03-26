# GRUPO 5 - CORHUILA - ASIGNATURA INTELIGENCIA ARTIFICIAL

## Integrantes
- Jhon Jamez Nieto Perez  
- Johan Sebastian Naranjo Quiroga  
- Juan Felipe Narvaez Amaya  
- Juan Jose Urbano Perdomo  

## Taller
Taller NLP (Corte 2 — Taller 1)

## Asignatura
Inteligencia Artificial

## Resumen
En el marco del Taller NLP de la asignatura *Inteligencia Artificial*, se desarrolló un **pipeline de Procesamiento del Lenguaje Natural (NLP)** orientado al análisis exploratorio de reseñas. El proyecto aborda: (1) **limpieza y preprocesamiento de texto** en español, (2) **extracción de entidades nombradas (NER)** con `spaCy`, (3) **clasificación de sentimiento** mediante un enfoque basado en **palabras clave**, (4) **descubrimiento de términos frecuentes por sentimiento** y (5) **agrupación de reseñas similares** usando representaciones TF-IDF y similitud coseno. Finalmente, el sistema genera un reporte en `txt` con resultados cuantitativos y estructurados.

Este informe documenta el objetivo del proyecto, las características técnicas del taller, el desarrollo metodológico y los resultados obtenidos con base en `Reseña.txt` y el reporte generado en `reporte_resultados.txt`.

---

## 1. Contexto y Objetivos

### 1.1 Objetivo General
Construir y documentar un pipeline de NLP que permita analizar un conjunto de reseñas en español mediante técnicas de preprocesamiento, extracción de entidades, análisis de sentimiento y agrupación por similitud textual.

### 1.2 Objetivos Específicos
1. Implementar funciones de **limpieza** y **normalización** del texto.
2. Aplicar **tokenización** y remoción de **stopwords** para reducir ruido lingüístico.
3. Extraer entidades nombradas (personas, organizaciones, lugares y entidades misceláneas) usando `spaCy`.
4. Calcular un **sentimiento** por reseña a partir de un diccionario de palabras positivas y negativas.
5. Obtener **palabras clave** por sentimiento (frecuencia de términos).
6. Agrupar reseñas según su **similitud** mediante TF-IDF y similitud coseno.
7. Generar un reporte final con métricas y hallazgos.

---

## 2. ¿De qué trata el proyecto?

El proyecto consiste en un sistema que toma un archivo de reseñas (`Reseña.txt`) donde **cada línea corresponde a una reseña en español**, y aplica un pipeline NLP para producir un reporte con información útil para análisis de opinión. El flujo general es:

1. **Lectura de reseñas:** se carga el contenido línea por línea.
2. **Limpieza del texto:** se eliminan URLs, correos y caracteres no alfabéticos (manteniendo letras con acentos).
3. **Preprocesamiento:** se tokeniza por separación en espacios y se eliminan *stopwords* en español.
4. **Extracción de entidades (NER):** con el modelo `es_core_news_sm` se identifican entidades con etiquetas `PER`, `ORG`, `LOC` y `MISC`.
5. **Análisis de sentimiento:** se asigna un puntaje sumando ocurrencias de palabras positivas y negativas; luego se clasifica como `positivo`, `negativo` o `neutral`.
6. **Palabras clave por sentimiento:** se consolidan los tokens por etiqueta y se reportan las más frecuentes.
7. **Agrupación de reseñas similares:** se calcula similitud coseno entre reseñas representadas por TF-IDF y se agrupan por un umbral configurado.
8. **Generación del reporte:** se escribe un archivo con el promedio de sentimiento, top palabras, entidades más mencionadas, agrupaciones y detalle por reseña.

---

## 3. Características Técnicas del Taller

### 3.1 Entrada y tipo de datos
- Entrada: archivo de texto (`Reseña.txt`) con reseñas en español.
- Formato: una reseña por línea.
- Salida: archivo de reporte en `txt` (`reporte_resultados.txt`).

### 3.2 Lenguaje
- Español (con soporte de stopwords y NER en idioma español).

### 3.3 Técnicas implementadas
1. **Limpieza textual**
   - Conversión a minúsculas.
   - Eliminación de URLs y correos.
   - Eliminación de caracteres especiales, conservando letras (incluyendo vocales acentuadas y `ñ`).
   - Normalización de espacios.
2. **Tokenización + Stopwords**
   - Tokenización basada en separación por espacios (tras limpieza).
   - Remoción de *stopwords* con `nltk.corpus.stopwords`.
3. **NER (Named Entity Recognition)**
   - Modelo `spaCy` en español (`es_core_news_sm`).
   - Filtrado de entidades por tipos relevantes: `PER`, `ORG`, `LOC`, `MISC`.
4. **Análisis de sentimiento heurístico**
   - Diccionarios de palabras positivas y negativas (p. ej., “excelente”, “amable”, “perfecto”; y “malo”, “horrible”, “deficiente”, “dañado”, entre otras).
   - Puntaje por conteo: si el puntaje es mayor que 0 -> `positivo`; menor que 0 -> `negativo`; si es 0 -> `neutral`.
5. **Palabras clave por sentimiento**
   - Consolidación de tokens filtrados por etiqueta.
   - Conteo con frecuencia para obtener el Top N.
6. **Similitud entre reseñas (agrupación)**
   - Representación con `TfidfVectorizer`.
   - Cálculo de similitud coseno (`cosine_similarity`).
   - Agrupación mediante umbral de similitud (en el código se utiliza `umbral=0.35`).

### 3.4 Librerías utilizadas
- `nltk` (stopwords).
- `spacy` (NER en español).
- `scikit-learn` (`TfidfVectorizer` y similitud coseno).

---

## 4. Metodología de Desarrollo

### 4.1 Preprocesamiento
El preprocesamiento se basa en dos etapas:
1. `limpiar_texto(texto)` elimina ruido (URLs, correos y caracteres especiales) y normaliza espacios.
2. `preprocesar_texto(texto)` tokeniza y elimina *stopwords* en español.

Con esto se busca:
- mejorar la calidad de los tokens utilizados en sentimiento y palabras clave,
- reducir ruido semántico causado por términos irrelevantes,
- mantener consistencia del texto (minúsculas y eliminación de caracteres no alfabéticos).

### 4.2 Extracción de entidades (NER)
Para cada reseña se ejecuta el modelo de `spaCy` y se extraen entidades con etiquetas:
- `PER` (personas)
- `ORG` (organizaciones)
- `LOC` (lugares)
- `MISC` (misceláneas)

El objetivo es identificar información relevante para el análisis (marcas, empresas, ciudades, etc.).

### 4.3 Análisis de sentimientos
El sentimiento se estima con un enfoque heurístico:
1. Se tokeniza y se filtra el texto (stopwords).
2. Se recorre cada token y se actualiza un `puntaje`:
   - tokens en `PALABRAS_POSITIVAS` suman,
   - tokens en `PALABRAS_NEGATIVAS` restan.
3. Se clasifica según el puntaje total:
   - `positivo` si el puntaje > 0
   - `negativo` si el puntaje < 0
   - `neutral` si el puntaje == 0

Este método es simple, interpretable y permite observar señales lingüísticas frecuentes asociadas a cada clase.

### 4.4 Agrupación por similitud
Para identificar reseñas similares se utiliza:
1. `TfidfVectorizer` para convertir el corpus a vectores TF-IDF.
2. `cosine_similarity` para medir similitud entre cada par de reseñas.
3. Agrupación por umbral (`umbral=0.35`) para reunir reseñas con alta cercanía textual.

---

## 5. Resultados

Los resultados principales se encuentran en `reporte_resultados.txt`. A continuación se sintetizan los hallazgos cuantitativos y descriptivos reportados.

### 5.1 Sentimiento promedio
- Sentimiento promedio: **0.31**

Este valor sugiere una tendencia global ligeramente positiva frente a neutral/negativa, coherente con la presencia de términos positivos con mayor frecuencia relativa.

### 5.2 Palabras clave por sentimiento (Top observables)

**Positivo (Top 10)**
- excelente: 8
- atención: 6
- rápido: 5
- amable: 5
- perfecto: 4
- compra: 4
- compré: 3

**Negativo (Top 10)**
- producto: 5
- llegó: 4
- además: 4
- soporte: 3
- horrible: 3
- terrible: 2
- experiencia: 2
- dañado: 2

**Neutral (Top 10)**
- llegó: 6
- bastante: 6
- experiencia: 4
- artículo: 4
- servicio: 3

### 5.3 Entidades más mencionadas
- entidades más mencionadas (Top observables):
  - llegó: 6
  - Bogotá: 2
  - Medellín: 2
  - Microsoft: 2
  - Google: 2
  - (y otras apariciones con menor frecuencia)

Estas entidades reflejan que las reseñas contienen referencias a lugares y organizaciones, además de patrones textuales repetidos (“llegó” aparece con alta frecuencia).

### 5.4 Agrupación de reseñas similares
El sistema agrupa reseñas similares según la similitud coseno sobre TF-IDF, utilizando un umbral configurado en el código (`umbral=0.35`). Este componente permite detectar subconjuntos de reseñas con patrones lingüísticos compartidos (por ejemplo, reseñas con quejas sobre entrega/soporte o reseñas que destacan rapidez y atención).

### 5.5 Detalle por reseña (trazabilidad)
El reporte incluye, para cada reseña:
- el texto,
- la etiqueta de sentimiento,
- el puntaje asociado,
- y la lista de entidades detectadas.

Esto mejora la interpretabilidad, ya que las conclusiones pueden relacionarse con casos concretos.

---

## 6. Análisis y Discusión

1. **Interpretabilidad:** el análisis de sentimiento basado en palabras clave facilita comprender por qué una reseña se clasifica como `positivo`, `negativo` o `neutral`.
2. **Calidad del preprocesamiento:** la eliminación de URLs/correos y la remoción de *stopwords* reducen el ruido y concentran los tokens en términos con potencial semántico.
3. **NER como complemento contextual:** las entidades detectadas con `spaCy` aportan contexto para entender *sobre qué* se está opinando (personas, organizaciones y lugares).
4. **Similitud textual vs. “misma intención”:** agrupar por similitud coseno identifica reseñas textualmente parecidas; sin embargo, puede agrupar reseñas con estructura similar aunque el sentimiento sea diferente, dependiendo de la forma en que estén escritas.

---

## 7. Dificultades Presentadas

1. **Sentimiento heurístico (limitaciones):** al usar un diccionario de palabras positivas/negativas, el sistema puede perder información cuando hay sarcasmo, negaciones complejas o expresiones implícitas.
2. **Variación lingüística:** la presencia de palabras no incluidas en el diccionario puede generar puntajes cercanos a cero y, por tanto, clasificaciones `neutral`.
3. **Umbral de similitud:** el umbral `0.35` influye directamente en el tamaño de los grupos formados; valores más altos reducen grupos (muestran similitudes más estrictas) y valores más bajos los aumentan.

---

## 8. Conclusiones

- Se implementó un pipeline NLP completo y documentado para el análisis de reseñas en español, integrando limpieza, preprocesamiento, NER, sentimiento y agrupación por similitud.
- El sistema generó resultados cuantificables en `reporte_resultados.txt`, destacando:
  - **sentimiento promedio de 0.31**,
  - palabras clave frecuentes por etiqueta (positivo, negativo y neutral),
  - y entidades más mencionadas en el corpus.
- La combinación de técnicas (stopwords + NER + TF-IDF) permite obtener un reporte estructurado, interpretable y útil para explorar opiniones reales.

---

## 9. Trabajo Futuro (Mejoras)

1. Sustituir o complementar el sentimiento heurístico con un modelo supervisado (por ejemplo, fine-tuning de un modelo Transformer en español) para mejorar precisión.
2. Mejorar manejo lingüístico:
   - detección de negaciones (“no fue…”, “no llegó…”),
   - reconocimiento de negación + polaridad.
3. Ajustar y evaluar el umbral de similitud TF-IDF (por ejemplo, mediante métricas de agrupación o validación manual).
4. Ampliar métricas del reporte (distribución por clase, análisis por fuente/entidad).

---

## 10. Referencias (base del enfoque)
- Documentación de `spaCy` (NER en español).
- Documentación de `nltk` (stopwords).
- Documentación de `scikit-learn` (`TfidfVectorizer` y `cosine_similarity`).


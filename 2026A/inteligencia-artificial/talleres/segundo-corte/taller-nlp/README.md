# Taller: Procesamiento del Lenguaje Natural (NLP)

## Descripcion

Taller completo de 10 ejercicios que cubre las tecnicas fundamentales del procesamiento del lenguaje natural, desde la limpieza basica de texto hasta un proyecto integrador de analisis de sentimientos.

## Contenido

1. **Limpieza y Preprocesamiento** - Remocion de URLs, emails, caracteres especiales y normalizacion de espacios
2. **Tokenizacion** - Division de textos en palabras y oraciones con NLTK
3. **Normalizacion de Texto** - Remocion de acentos y caracteres diacriticos con `unicodedata`
4. **Remocion de Stopwords** - Filtrado de palabras comunes sin significado semantico
5. **Stemming vs Lemmatizacion** - Comparacion de tecnicas de reduccion de palabras
6. **Analisis de Frecuencia** - Identificacion de palabras clave mas frecuentes con TF-IDF
7. **Extraccion de Entidades (NER)** - Reconocimiento de personas, lugares y organizaciones con spaCy
8. **Similitud entre Textos** - Calculo de similitud coseno con TF-IDF
9. **Pipeline Completo** - Clase `PreprocessadorTexto` que integra todas las tecnicas
10. **Proyecto Integrador** - Analisis de sentimientos con preprocesamiento, NER y agrupacion de resenas

## Dependencias

- `nltk`
- `scikit-learn`
- `spacy` (modelo `es_core_news_sm`)
- `numpy`
- `re`, `unicodedata` (stdlib)

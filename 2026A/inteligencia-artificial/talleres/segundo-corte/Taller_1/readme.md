## 🤖 Taller 1 - Procesamiento de Lenguaje Natural (NLP)  

---

## 📂 Archivo del Taller

Puedes acceder al documento completo aquí:

📄 [Ver Taller 1](./Taller%201.pdf)

---

## 📌 Introducción  

En el presente taller se aborda el estudio del **Procesamiento del Lenguaje Natural (NLP)**, enfocándose en la aplicación de técnicas fundamentales para el análisis y manipulación de textos. A través de diferentes ejercicios prácticos, se busca comprender cómo las máquinas pueden procesar, interpretar y extraer información del lenguaje humano.  

El desarrollo incluye procesos como la limpieza y preprocesamiento de texto, tokenización, normalización, eliminación de palabras vacías (*stopwords*), así como técnicas más avanzadas como *stemming*, lematización, análisis de frecuencia, extracción de entidades nombradas (NER) y cálculo de similitud entre textos.  

Además, se integra todo este conocimiento en la construcción de un *pipeline* completo de procesamiento y un proyecto final orientado al análisis de sentimientos.  

---

## 🎯 Objetivo General  

Desarrollar e implementar técnicas fundamentales de Procesamiento del Lenguaje Natural (NLP) que permitan el análisis, procesamiento e interpretación de textos, mediante la construcción de un pipeline completo y su aplicación en un proyecto de análisis de sentimientos.  

---

## 🎯 Objetivos Específicos  

- Implementar procesos de limpieza y preprocesamiento de texto  
- Aplicar técnicas de tokenización en palabras y oraciones  
- Normalizar texto eliminando acentos y caracteres especiales  
- Remover palabras vacías (*stopwords*)  
- Comparar *stemming* y lematización  
- Analizar la frecuencia de palabras en textos  
- Extraer entidades nombradas (NER)  
- Calcular similitud entre textos con TF-IDF y coseno  
- Construir un pipeline completo de procesamiento  
- Aplicar todo en un análisis de sentimientos  

---

## 🧠 Marco Teórico  

El **Procesamiento del Lenguaje Natural (NLP)** es una rama de la inteligencia artificial que permite a las computadoras comprender, interpretar y generar lenguaje humano.  

El proceso inicia con el **preprocesamiento de texto**, donde se limpian datos eliminando elementos irrelevantes como URLs, correos y caracteres especiales. Luego se aplica la **tokenización**, que divide el texto en unidades más pequeñas llamadas tokens.  

La **normalización** reduce la variabilidad del lenguaje eliminando acentos y estandarizando el texto. También se eliminan las **stopwords**, palabras comunes que no aportan significado relevante.  

Para simplificar palabras se utilizan técnicas como el **stemming**, que reduce palabras a su raíz, y la **lematización**, que obtiene su forma base correcta.  

El análisis de texto incluye también el **análisis de frecuencia**, que identifica palabras más relevantes, y la **extracción de entidades nombradas (NER)**, que permite reconocer nombres de personas, lugares y organizaciones.  

Además, se emplean técnicas como **TF-IDF** y **similitud coseno** para medir qué tan similares son dos textos.  

Finalmente, todas estas técnicas se integran en un **pipeline de procesamiento**, permitiendo automatizar tareas como el **análisis de sentimientos**, donde los textos se clasifican como positivos, negativos o neutrales.  

---

## ⚙️ Desarrollo del Taller  

### 🔵 5.1 Limpieza y Preprocesamiento  
Eliminación de URLs, correos, caracteres especiales y normalización de texto.  

---

### 🔵 5.2 Tokenización  
División de textos en palabras y oraciones utilizando NLTK.  

---

### 🔵 5.3 Normalización  
Eliminación de acentos y caracteres diacríticos.  

---

### 🔵 5.4 Stopwords  
Filtrado de palabras sin valor semántico.  

---

### 🔵 5.5 Stemming vs Lematización  
Comparación entre reducción de palabras por raíz y por forma base.  

---

### 🔵 5.6 Frecuencia de Palabras  
Identificación de palabras más comunes en el texto.  

---

### 🔵 5.7 NER  
Extracción de entidades como personas, lugares y organizaciones.  

---

### 🔵 5.8 Similaridad de Textos  
Cálculo de similitud usando TF-IDF y coseno.  

---

### 🔵 5.9 Pipeline  
Integración de todas las técnicas en una sola clase de procesamiento.  

---

### 🔵 5.10 Proyecto Integrador
Aplicación en análisis de sentimientos (positivo, negativo, neutral).  

---

## 💻 Implementación Práctica  

- 📓 Procesamiento NLP (Notebook)  
- 🐍 Scripts en Python  
- Uso de librerías: NLTK, spaCy, scikit-learn  

> 💡 Se integran todas las técnicas en un pipeline funcional aplicado a análisis de sentimientos.  

---

## ✅ Conclusiones  

El taller permitió aplicar de manera práctica los principales conceptos del Procesamiento del Lenguaje Natural, evidenciando su utilidad en el análisis de datos textuales.  

Se implementaron correctamente técnicas como limpieza, tokenización, normalización, eliminación de stopwords y reducción de palabras. Además, se aplicaron métodos más avanzados como NER, análisis de frecuencia y similitud de textos.  

La construcción de un pipeline permitió integrar todos los procesos de manera estructurada, y el proyecto final de análisis de sentimientos permitió aplicar los conocimientos en un caso real.  

---

## 📚 Bibliografía  

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*  
- Bishop, C. *Pattern Recognition and Machine Learning*  
- Haykin, S. *Neural Networks and Learning Machines*  
- Géron, A. *Hands-On Machine Learning*  
- Goodfellow, I. et al. *Deep Learning*  

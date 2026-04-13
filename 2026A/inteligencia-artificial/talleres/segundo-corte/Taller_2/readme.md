# 🤖 Taller 2 Representaciones de Texto - Word Embeddings y Embedding Contextual

---

## 📂 Archivo del Taller

Puedes acceder al documento completo aquí:

📄 [Ver Taller ](./Taller.pdf)

---

## 📌 Introducción  

En el presente taller se aborda el estudio de las representaciones de texto en el campo de la Inteligencia Artificial, haciendo énfasis en el uso de word embeddings y embeddings contextuales como herramientas fundamentales para el procesamiento del lenguaje natural. Estas técnicas permiten transformar palabras en vectores numéricos que capturan relaciones semánticas y sintácticas, facilitando que los modelos computacionales puedan interpretar y analizar el lenguaje humano de manera más eficiente.

A lo largo del desarrollo del taller, se implementaron diferentes actividades prácticas que incluyen la comparación entre representaciones tradicionales como One-Hot Encoding y representaciones densas, el entrenamiento de modelos como Word2Vec, así como la visualización de embeddings mediante técnicas de reducción de dimensionalidad como t-SNE. Además, se exploró la aplicación de estas representaciones en tareas reales como la clasificación de textos utilizando modelos de aprendizaje automático.

El objetivo principal de este trabajo fue comprender no solo el funcionamiento teórico de los embeddings, sino también su aplicación práctica en problemas reales, permitiendo evidenciar su importancia en el desarrollo de sistemas inteligentes basados en lenguaje.

---

## 🎯 Objetivo General  

Comprender el uso de representaciones vectoriales de palabras (word embeddings) en el procesamiento del lenguaje natural, mediante la implementación de modelos como Word2Vec y la visualización de sus relaciones semánticas utilizando técnicas de reducción de dimensionalidad como t-SNE, con el fin de analizar cómo la inteligencia artificial interpreta el lenguaje humano.

---

## 🎯 Objetivos Específicos  

- Implementar representaciones de palabras mediante modelos de embeddings como Word2Vec para transformar texto en vectores numéricos.
- Seleccionar palabras representativas del modelo entrenado y extraer sus vectores para su análisis.
- Aplicar la técnica de reducción de dimensionalidad t-SNE para visualizar los embeddings en un espacio bidimensional.
- Interpretar gráficamente las relaciones semánticas entre palabras según su cercanía en el espacio vectorial.
- Analizar la utilidad de los embeddings en tareas de procesamiento de lenguaje natural y clasificación de texto.

---

## 🧠 Marco Teórico  

El procesamiento del lenguaje natural (PLN) es una rama de la inteligencia artificial que se enfoca en la interacción entre las computadoras y el lenguaje humano. Uno de los principales retos en este campo es la representación del texto de manera que los modelos computacionales puedan interpretarlo eficientemente.

Tradicionalmente, técnicas como el One-Hot Encoding han sido utilizadas para representar palabras como vectores binarios, donde cada palabra se identifica con una posición única en un vector de alta dimensión. Sin embargo, este enfoque presenta limitaciones importantes, como la alta dispersión de los datos y la incapacidad de capturar relaciones semánticas entre palabras.

Para superar estas limitaciones, surgen los Word Embeddings, que son representaciones vectoriales densas en espacios de menor dimensión. Estos embeddings permiten capturar similitudes semánticas y relaciones sintácticas entre palabras. Modelos como Word2Vec, basado en redes neuronales, aprenden estas representaciones a partir de grandes corpus de texto, logrando que palabras con significados similares tengan vectores cercanos en el espacio vectorial.

Además, existen técnicas de visualización como t-SNE (t-Distributed Stochastic Neighbor Embedding), que permiten reducir la dimensionalidad de los embeddings para representarlos en espacios bidimensionales o tridimensionales, facilitando su interpretación.

En aplicaciones prácticas, los embeddings son ampliamente utilizados en tareas como clasificación de textos, análisis de sentimientos, traducción automática y sistemas de recomendación. Su capacidad para capturar el contexto y significado del lenguaje los convierte en una herramienta fundamental en el desarrollo de sistemas inteligentes modernos.

Finalmente, los avances recientes han dado lugar a los embeddings contextuales, utilizados en modelos más complejos como los basados en transformadores, los cuales consideran el contexto completo de una palabra dentro de una oración, mejorando significativamente el rendimiento en tareas de PLN.

---

## ⚙️ Desarrollo del Taller  

-5.1 Instalación de Dependencias
-5.2 Conceptos Fundamentales
-5.3 Word2Vec - Entrenamiento Práctico
-5.4 Visualización con t-SNE
-5.5 Clasificación de Textos con Embeddings

---

## ✅ Conclusiones  

El desarrollo del presente taller permitió aplicar de manera práctica los principales conceptos y técnicas del Procesamiento del Lenguaje Natural (NLP), evidenciando su utilidad en el análisis y tratamiento de datos textuales. A través de la ejecución de cada uno de los ejercicios, se logró implementar correctamente procesos como la limpieza y normalización de texto, la tokenización, la eliminación de stopwords y la reducción de palabras mediante stemming y lematización.

Asimismo, se adquirió experiencia en técnicas más avanzadas como el análisis de frecuencia de palabras, la extracción de entidades nombradas (NER) y el cálculo de similitud entre textos, lo que permitió comprender cómo se puede extraer información relevante y medir relaciones semánticas entre diferentes contenidos. La construcción de un pipeline completo de preprocesamiento facilitó la integración de todas estas etapas, demostrando la importancia de estructurar adecuadamente el flujo de procesamiento de datos.

Finalmente, el desarrollo del proyecto integrador de análisis de sentimientos permitió consolidar los conocimientos adquiridos, aplicándolos en un caso práctico que simula situaciones reales. En conjunto, este trabajo fortaleció tanto las habilidades técnicas en programación como la comprensión de cómo las máquinas pueden interpretar el lenguaje humano, resaltando el papel fundamental del NLP en el campo de la inteligencia artificial.

---

## 📚 Bibliografía  

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*  
- Bishop, C. *Pattern Recognition and Machine Learning*  
- Haykin, S. *Neural Networks and Learning Machines*  
- Géron, A. *Hands-On Machine Learning*  
- Goodfellow, I. et al. *Deep Learning*  

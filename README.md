# Sistema Avanzado de Corrección de Apellidos

Este proyecto utiliza técnicas avanzadas de machine learning y procesamiento de lenguaje natural para detectar y corregir apellidos mal escritos en documentos Word, comparándolos con una base de datos de referencia de apellidos correctos.

## Descripción General

El sistema implementa un enfoque de dos etapas para la corrección de apellidos:

1. **Detección de Apellidos**: Utiliza Named Entity Recognition (NER) y reglas heurísticas para identificar palabras que probablemente sean apellidos.
2. **Corrección de Apellidos**: Para cada apellido detectado, utiliza un modelo de machine learning entrenado para determinar la versión correcta del apellido.

## Características Principales

- **Detección avanzada de apellidos**:
  - Reconocimiento de entidades nombradas (NER) con spaCy
  - Reglas heurísticas basadas en contexto
  - Listas de exclusión para lugares y palabras comunes
  - Modelo de clasificación para determinar si una palabra es apellido

- **Corrección precisa de apellidos**:
  - Modelos de machine learning (Random Forest, SVM, etc.)
  - Características basadas en similitud de texto (Levenshtein, Soundex, n-gramas)
  - Umbral de confianza configurable para evitar falsos positivos
  
- **Procesamiento de documentos Word**:
  - Extracción de texto de documentos Word
  - Aplicación de correcciones manteniendo el formato
  - Generación de informes detallados de correcciones

- **Flexibilidad y configuración**:
  - Umbrales de confianza ajustables para detección y corrección
  - Opciones para habilitar/deshabilitar NER
  - Listas de exclusión personalizables

## Instalación

1. Instalar dependencias:
```
pip install -r requirements.txt
```

2. Instalar el modelo español de spaCy para NER:
```
python -m spacy download es_core_news_md
```

3. Preparar los datos:
   - Coloque su archivo CSV o Excel con apellidos correctos en el directorio `data/`
   - Coloque sus documentos Word en el directorio `data/documents/`
   - Opcionalmente, prepare listas de exclusión en `data/exclusion/`

## Uso

### Procesamiento de documentos Word

```
python word_service.py --model-path models/surname_correction_model.pkl \
                      --reference-data data/processed_surnames.csv \
                      --input-file data/documents/documento.docx \
                      --output-dir data/corrected_word \
                      --confidence-threshold 0.7 \
                      --surname-confidence-threshold 0.6 \
                      --use-ner
```

### Entrenamiento de modelos

1. Entrenar el modelo de corrección:
```
python train_correction_model.py --data data/training_data.csv --output models/surname_correction_model.pkl
```

2. Entrenar el modelo de detección de apellidos (opcional):
```
python train_surname_detection.py --data data/labeled_words.csv --output models/surname_detection_model.pkl
```

## Estructura del Proyecto

- `data/` - Directorio para archivos de datos
  - `reference/` - Datos de referencia con apellidos correctos
  - `documents/` - Documentos Word para procesar
  - `corrected_word/` - Documentos corregidos y reportes
  - `exclusion/` - Listas de exclusión (lugares, palabras comunes)
  
- `src/` - Código fuente
  - `data_processing.py` - Funciones para carga y preprocesamiento de datos
  - `feature_engineering.py` - Extracción de características para modelos ML
  - `models.py` - Implementación de modelos de machine learning y clase SurnameCorrector
  - `evaluation.py` - Métricas y funciones de evaluación
  - `surname_detection.py` - Módulo para detección avanzada de apellidos con NER y reglas
  
- `models/` - Modelos entrenados
  - `surname_correction_model.pkl` - Modelo para corrección de apellidos
  - `surname_detection_model.pkl` - Modelo para detección de apellidos (opcional)
  
- `word_service.py` - Servicio para procesar documentos Word
- `train_correction_model.py` - Script para entrenar el modelo de corrección
- `train_surname_detection.py` - Script para entrenar el modelo de detección de apellidos
- `notebooks/` - Jupyter notebooks para exploración y visualización

## Arquitectura del Sistema

### Enfoque de Dos Etapas

1. **Detección de Apellidos**:
   - Utiliza NER para identificar entidades de tipo PERSONA
   - Aplica reglas heurísticas basadas en contexto
   - Filtra falsos positivos usando listas de exclusión
   - Opcionalmente utiliza un clasificador ML entrenado

2. **Corrección de Apellidos**:
   - Extrae características de similitud para cada candidato
   - Utiliza un modelo Random Forest para predecir la corrección
   - Aplica un umbral de confianza para evitar correcciones incorrectas

### Flujo de Procesamiento de Documentos

1. Carga del documento Word y extracción de texto
2. Detección de potenciales apellidos en el texto
3. Para cada apellido detectado, generación de candidatos de corrección
4. Predicción de la corrección más probable con nivel de confianza
5. Aplicación de correcciones al documento original
6. Generación de informe detallado de correcciones

## Notas Adicionales

- El sistema está diseñado para minimizar falsos positivos, priorizando la precisión sobre la exhaustividad.
- Las listas de exclusión son fundamentales para evitar la corrección de lugares o palabras comunes.
- El uso de NER mejora significativamente la detección de apellidos en contexto.
- Los umbrales de confianza son ajustables según las necesidades específicas de cada caso.

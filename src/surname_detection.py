"""
Módulo para la detección avanzada de apellidos utilizando NER y reglas contextuales.
"""

import re
import os
import logging
import spacy
from typing import List, Dict, Tuple, Optional

# Configurar logging
logger = logging.getLogger('surname_detection')

# Cargar modelo de spaCy para NER
nlp = None
try:
    nlp = spacy.load("en_core_web_md")
    logger.info("Modelo de spaCy en inglés cargado correctamente")
except Exception as e:
    logger.warning(f"No se pudo cargar el modelo de spaCy: {e}")
    logger.warning("Se utilizará un enfoque basado en reglas únicamente")

# Listas de exclusión
COMMON_PLACES = set([
    "madrid", "barcelona", "valencia", "sevilla", "zaragoza", "málaga", "murcia", 
    "palma", "bilbao", "alicante", "córdoba", "valladolid", "vigo", "gijón", 
    "españa", "francia", "alemania", "italia", "portugal", "reino unido", "estados unidos",
    "north", "south", "east", "west", "central", "dist", "district"
])

COMMON_WORDS = set([
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "porque",
    "aunque", "si", "no", "como", "cuando", "donde", "quien", "cuyo", "cuya",
    "january", "february", "march", "april", "may", "june", "july", "august", 
    "september", "october", "november", "december",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre"
])

# Prefijos comunes que suelen preceder a apellidos
SURNAME_PREFIXES = [
    'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Sra.', 'Srta.', 'D.', 'Dña.',
    'Don', 'Doña', 'Doctor', 'Doctora', 'Profesor', 'Profesora'
]

def load_exclusion_lists(places_file: Optional[str] = None, words_file: Optional[str] = None) -> Tuple[set[str], set[str]]:
    """
    Carga listas de exclusión para mejorar la detección de apellidos.
    
    Args:
        places_file: Ruta al archivo con nombres de lugares
        words_file: Ruta al archivo con palabras comunes
        
    Returns:
        Tupla de (lugares comunes, palabras comunes)
    """
    common_places = set()
    common_words = set()
    
    # Usar rutas predeterminadas si no se proporcionan
    places_path = places_file or 'data/common_places.txt'
    words_path = words_file or 'data/common_words.txt'
    
    # Intentar cargar lista de lugares
    try:
        with open(places_path, 'r', encoding='utf-8') as f:
            common_places = set(line.strip().lower() for line in f if line.strip())
        logger.info(f"Cargados {len(common_places)} lugares comunes desde {places_path}")
    except Exception as e:
        logger.warning(f"No se pudo cargar la lista de lugares desde {places_path}: {e}")
    
    # Intentar cargar lista de palabras comunes
    try:
        with open(words_path, 'r', encoding='utf-8') as f:
            common_words = set(line.strip().lower() for line in f if line.strip())
        logger.info(f"Cargadas {len(common_words)} palabras comunes desde {words_path}")
    except Exception as e:
        logger.warning(f"No se pudo cargar la lista de palabras comunes desde {words_path}: {e}")
    
    return common_places, common_words

def is_likely_surname(word: str, text_context: str, common_places: set[str], common_words: set[str]) -> bool:
    """
    Determina si una palabra es probablemente un apellido basado en múltiples criterios.
    
    Args:
        word: Palabra candidata a ser apellido
        text_context: Texto circundante para análisis contextual
        common_places: Conjunto de nombres de lugares comunes (para exclusión)
        common_words: Conjunto de palabras comunes (para exclusión)
        
    Returns:
        True si es probable que sea un apellido, False en caso contrario
    """
    # Ignorar palabras vacías o muy cortas
    if not word or len(word) < 2:
        return False
    
    # Verificar si es un lugar conocido o palabra común (no apellido)
    if word.lower() in common_places or word.lower() in common_words:
        return False
    
    # Si no comienza con mayúscula, probablemente no es un apellido
    if not word[0].isupper():
        return False
    
    # Verificar contexto (prefijos típicos de apellidos)
    for prefix in SURNAME_PREFIXES:
        if prefix + ' ' + word in text_context or prefix + '. ' + word in text_context:
            return True
    
    # Verificar patrones típicos de nombres completos (Nombre Apellido)
    name_pattern = r'\b[A-Z][a-z]+ ' + re.escape(word) + r'\b'
    if re.search(name_pattern, text_context):
        return True
    
    return False

def extract_surnames_with_ner(text: str) -> List[Tuple[str, str]]:
    """
    Extrae apellidos usando Named Entity Recognition de spaCy.
    
    Args:
        text: Texto para analizar
        
    Returns:
        Lista de tuplas (apellido, contexto)
    """
    if not nlp:
        logger.warning("Modelo de spaCy no disponible, no se pueden extraer entidades")
        return []
    
    surnames = []
    try:
        doc = nlp(text)
        
        # Extraer entidades de tipo PERSON (en inglés es PERSON, no PER)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Intentar extraer el apellido de la entidad persona
                name_parts = ent.text.split()
                if len(name_parts) > 1:
                    # Asumimos que el último token es el apellido
                    surname = name_parts[-1]
                    # Obtener contexto alrededor de la entidad
                    start = max(0, ent.start_char - 20)
                    end = min(len(text), ent.end_char + 20)
                    context = text[start:end]
                    surnames.append((surname, context))
    except Exception as e:
        logger.error(f"Error al procesar texto con spaCy: {e}")
        return []
    
    return surnames

def extract_potential_surnames(text: str, use_ner: bool = True, 
                         places_file: Optional[str] = None, words_file: Optional[str] = None,
                         common_places: Optional[set[str]] = None, common_words: Optional[set[str]] = None) -> List[Tuple[str, str]]:
    """
    Extrae potenciales apellidos del texto usando una combinación de NER y reglas.
    
    Args:
        text: Texto para analizar
        use_ner: Si es True, utiliza NER además de reglas
        places_file: Ruta al archivo con nombres de lugares
        words_file: Ruta al archivo con palabras comunes
        common_places: Conjunto de lugares comunes (opcional)
        common_words: Conjunto de palabras comunes (opcional)
        
    Returns:
        Lista de tuplas (apellido, contexto)
    """
    # Cargar listas de exclusión si no se proporcionaron
    if common_places is None or common_words is None:
        loaded_places, loaded_words = load_exclusion_lists(places_file, words_file)
        common_places = common_places or loaded_places
        common_words = common_words or loaded_words
    
    # Resultados combinados
    potential_surnames = []
    
    # Usar NER si está disponible y habilitado
    if use_ner and nlp:
        ner_surnames = extract_surnames_with_ner(text)
        potential_surnames.extend(ner_surnames)
    
    # Aplicar enfoque basado en reglas
    # 1. Buscar palabras después de prefijos comunes
    for prefix in SURNAME_PREFIXES:
        pattern = prefix + r'\.?\s+([A-Z][a-zA-Z\-]+)'
        matches = re.finditer(pattern, text)
        for match in matches:
            surname = match.group(1)
            context = text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
            if is_likely_surname(surname, context, common_places, common_words):
                potential_surnames.append((surname, context))
    
    # 2. Buscar patrones de nombres completos
    name_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-zA-Z\-]+)\b'
    matches = re.finditer(name_pattern, text)
    for match in matches:
        first_name = match.group(1)
        surname = match.group(2)
        context = text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
        if is_likely_surname(surname, context, common_places, common_words):
            potential_surnames.append((surname, context))
    
    # Eliminar duplicados preservando el orden
    seen = set()
    unique_surnames = []
    for surname, context in potential_surnames:
        if surname not in seen:
            seen.add(surname)
            unique_surnames.append((surname, context))
    
    return unique_surnames

def filter_valid_surnames(potential_surnames: List[Tuple[str, str]], reference_surnames: List[str], 
                         similarity_threshold: float = 0.8) -> List[Tuple[str, str]]:
    """
    Filtra los apellidos potenciales para mantener solo los que son probablemente válidos.
    
    Args:
        potential_surnames: Lista de tuplas (apellido, contexto)
        reference_surnames: Lista de apellidos de referencia
        similarity_threshold: Umbral de similitud para considerar un apellido como válido
        
    Returns:
        Lista filtrada de tuplas (apellido, contexto)
    """
    from rapidfuzz import process
    
    valid_surnames = []
    
    for surname, context in potential_surnames:
        # Verificar si el apellido está en la lista de referencia o es similar
        if surname in reference_surnames:
            valid_surnames.append((surname, context))
            continue
        
        # Buscar el apellido más similar en la lista de referencia
        best_match, score = process.extractOne(surname, reference_surnames)
        if score / 100.0 >= similarity_threshold:
            valid_surnames.append((surname, context))
    
    return valid_surnames

"""
Script para probar la carga del modelo y la clase SurnameCorrector.
"""

import os
import sys
import pickle
import traceback
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Asegurarnos de que podemos importar los módulos del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_engineering import extract_features
from src.data_processing import load_reference_data

def main():
    """Función principal para probar la carga del modelo y la clase SurnameCorrector."""
    try:
        # Rutas de archivos
        model_path = 'models/random_forest_model.pkl'
        reference_data_path = 'data/processed_surnames.csv'
        
        print(f"Verificando si el modelo existe: {os.path.exists(model_path)}")
        print(f"Verificando si los datos de referencia existen: {os.path.exists(reference_data_path)}")
        
        # Cargar el modelo
        print("\nCargando modelo...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Tipo de modelo: {type(model).__name__}")
        print(f"Atributos del modelo: {dir(model)[:10]}...")
        
        # Verificar si el modelo tiene los métodos necesarios
        has_predict = hasattr(model, 'predict')
        has_predict_proba = hasattr(model, 'predict_proba')
        print(f"¿Tiene método predict?: {has_predict}")
        print(f"¿Tiene método predict_proba?: {has_predict_proba}")
        
        # Cargar datos de referencia
        print("\nCargando datos de referencia...")
        reference_df = pd.read_csv(reference_data_path)
        print(f"Columnas en los datos de referencia: {reference_df.columns.tolist()}")
        print(f"Número de registros: {len(reference_df)}")
        
        # Extraer apellidos de referencia
        print("\nExtrayendo apellidos de referencia...")
        if 'CleanedSurname' in reference_df.columns:
            all_surnames = reference_df['CleanedSurname'].dropna().unique().tolist()
        elif 'ExtractedSurname' in reference_df.columns:
            all_surnames = reference_df['ExtractedSurname'].dropna().unique().tolist()
        else:
            print("ERROR: No se encontró una columna de apellidos en los datos de referencia")
            return 1
        
        print(f"Número de apellidos únicos: {len(all_surnames)}")
        print(f"Ejemplos de apellidos: {all_surnames[:5]}")
        
        # Crear la clase SurnameCorrector
        print("\nCreando la clase SurnameCorrector...")
        
        class SurnameCorrector:
            """Clase para corregir apellidos usando un modelo de machine learning."""
            
            def __init__(self, model, reference_surnames: List[str]):
                """
                Inicializa el corrector de apellidos.
                
                Args:
                    model: Modelo entrenado (debe tener métodos predict y predict_proba)
                    reference_surnames: Lista de apellidos de referencia
                """
                self.model = model
                self.reference_surnames = reference_surnames
                print(f"SurnameCorrector inicializado con {len(reference_surnames)} apellidos de referencia")
            
            def correct_surname(self, surname: str) -> tuple:
                """
                Corrige un apellido utilizando el modelo entrenado.
                
                Args:
                    surname: Apellido a corregir
                    
                Returns:
                    Tupla (apellido_corregido, confianza)
                """
                # Si el apellido está vacío o es muy corto, devolverlo sin cambios
                if not surname or len(surname) < 3:
                    return surname, 0.0
                
                # Extraer características del apellido
                features = extract_features(surname)
                
                # Convertir a matriz numpy
                X = np.array([features])
                
                # Obtener predicciones y probabilidades
                try:
                    # Predecir la clase (índice del apellido correcto)
                    pred_idx = self.model.predict(X)[0]
                    
                    # Obtener probabilidades
                    probs = self.model.predict_proba(X)[0]
                    
                    # Obtener la confianza (probabilidad de la clase predicha)
                    confidence = probs[pred_idx]
                    
                    # Obtener el apellido corregido
                    corrected = self.reference_surnames[pred_idx]
                    
                    print(f"Apellido: {surname} -> Corregido: {corrected} (Confianza: {confidence:.2f})")
                    return corrected, confidence
                    
                except Exception as e:
                    print(f"Error al predecir: {e}")
                    traceback.print_exc()
                    return surname, 0.0
        
        # Crear una instancia del corrector
        print("\nCreando instancia del corrector...")
        corrector = SurnameCorrector(model, all_surnames)
        
        # Probar con algunos apellidos
        print("\nProbando corrección de apellidos:")
        test_surnames = ["Smth", "Jonson", "Brwn", "Mller", "Davs"]
        
        for surname in test_surnames:
            corrected, confidence = corrector.correct_surname(surname)
            print(f"{surname} -> {corrected} (Confianza: {confidence:.2f})")
        
        print("\nPrueba completada con éxito!")
        return 0
        
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

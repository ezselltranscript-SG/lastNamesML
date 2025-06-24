"""
Models module for surname correction project.
Implements various machine learning models for surname correction.
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaseModel:
    """Base class for surname correction models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            model_path: Path to saved model file (for loading)
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            preds = self.predict(X)
            probs = np.zeros((X.shape[0], 2))
            probs[np.arange(X.shape[0]), preds.astype(int)] = 1
            return probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }
    
    def save(self, model_path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, model_path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            model_path: Path to the saved model
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)


class RandomForestModel(BaseModel):
    """Random Forest model for surname correction."""
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the Random Forest model.
        
        Args:
            model_path: Path to saved model file (for loading)
            **kwargs: Additional arguments for RandomForestClassifier
        """
        super().__init__(model_path)
        self.kwargs = kwargs
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            **self.kwargs
        )
        self.model.fit(X, y)


class SVMModel(BaseModel):
    """Support Vector Machine model for surname correction."""
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the SVM model.
        
        Args:
            model_path: Path to saved model file (for loading)
            **kwargs: Additional arguments for SVC
        """
        super().__init__(model_path)
        self.kwargs = kwargs
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        self.model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42,
            **self.kwargs
        )
        self.model.fit(X, y)


class NeuralNetworkModel(BaseModel):
    """Neural Network model for surname correction."""
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the Neural Network model.
        
        Args:
            model_path: Path to saved model file (for loading)
            **kwargs: Additional arguments for MLPClassifier
        """
        super().__init__(model_path)
        self.kwargs = kwargs
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Neural Network model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
            **self.kwargs
        )
        self.model.fit(X, y)


class SurnameCorrector:
    """Main class for surname correction using machine learning with improved detection and correction."""
    
    def __init__(self, model_type: str = 'random_forest', model_path: Optional[str] = None,
                is_surname_model_path: Optional[str] = None):
        """
        Initialize the surname corrector with two-stage classification.
        
        Args:
            model_type: Type of model to use ('random_forest', 'svm', or 'neural_network')
            model_path: Path to saved correction model file
            is_surname_model_path: Path to saved surname detection model file
        """
        self.model_type = model_type
        self.correction_model = self._create_model(model_type, model_path)
        
        # Modelo para determinar si una palabra es un apellido (primera etapa)
        if is_surname_model_path and os.path.exists(is_surname_model_path):
            self.is_surname_model = load_model(is_surname_model_path)
            self.has_surname_classifier = True
        else:
            self.is_surname_model = None
            self.has_surname_classifier = False
            
        self.all_surnames = []
        self.common_places = set()
        self.common_words = set()
        
        # Configuración para NER
        self.use_ner = True
    
    def _create_model(self, model_type: str, model_path: Optional[str] = None) -> BaseModel:
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create
            model_path: Path to saved model file (for loading)
            
        Returns:
            Model instance
        """
        if model_type == 'random_forest':
            return RandomForestModel(model_path)
        elif model_type == 'svm':
            return SVMModel(model_path)
        elif model_type == 'neural_network':
            return NeuralNetworkModel(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the correction model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        self.correction_model.train(X, y)
    
    def train_surname_classifier(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the surname detection classifier.
        
        Args:
            X: Feature matrix (características de palabras)
            y: Target labels (1 si es apellido, 0 si no)
        """
        from sklearn.ensemble import RandomForestClassifier
        
        self.is_surname_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        self.is_surname_model.fit(X, y)
        self.has_surname_classifier = True
    
    def set_reference_surnames(self, surnames: List[str]) -> None:
        """
        Set the reference list of correct surnames.
        
        Args:
            surnames: List of correct surnames
        """
        self.all_surnames = surnames
    
    def set_exclusion_lists(self, common_places: set[str], common_words: set[str]) -> None:
        """
        Set the exclusion lists for better surname detection.
        
        Args:
            common_places: Set of common place names (not surnames)
            common_words: Set of common words (not surnames)
        """
        self.common_places = common_places
        self.common_words = common_words
    
    def is_surname(self, word: str, context: str = "") -> Tuple[bool, float]:
        """
        Determine if a word is likely a surname using the trained classifier or rules.
        
        Args:
            word: Word to check
            context: Surrounding text context
            
        Returns:
            Tuple of (is_surname, confidence)
        """
        from src.surname_detection import is_likely_surname
        
        # Si tenemos un clasificador entrenado, usarlo
        if self.has_surname_classifier and self.is_surname_model:
            # Extraer características relevantes para determinar si es apellido
            features = self._extract_surname_features(word, context)
            X = np.array([list(features.values())])
            
            # Predecir si es apellido
            is_surname_pred = self.is_surname_model.predict(X)[0]
            confidence = max(self.is_surname_model.predict_proba(X)[0])
            
            return bool(is_surname_pred), float(confidence)
        
        # Si no hay clasificador, usar reglas heurísticas
        is_surname_by_rules = is_likely_surname(word, context, self.common_places, self.common_words)
        return is_surname_by_rules, 0.8 if is_surname_by_rules else 0.2
    
    def _extract_surname_features(self, word: str, context: str) -> Dict[str, float]:
        """
        Extract features to determine if a word is a surname.
        
        Args:
            word: Word to extract features for
            context: Surrounding text
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Características básicas de la palabra
        features['length'] = len(word)
        features['starts_uppercase'] = 1.0 if word and word[0].isupper() else 0.0
        features['all_uppercase'] = 1.0 if word.isupper() else 0.0
        features['has_digits'] = 1.0 if any(c.isdigit() for c in word) else 0.0
        features['has_special'] = 1.0 if any(not c.isalnum() for c in word) else 0.0
        
        # Características contextuales
        from src.surname_detection import SURNAME_PREFIXES
        features['has_prefix'] = 0.0
        for prefix in SURNAME_PREFIXES:
            if prefix + ' ' + word in context or prefix + '. ' + word in context:
                features['has_prefix'] = 1.0
                break
        
        # Características de similitud con apellidos conocidos
        if self.all_surnames:
            from rapidfuzz import process
            best_match, score = process.extractOne(word, self.all_surnames)
            features['max_similarity'] = score / 100.0
        else:
            features['max_similarity'] = 0.0
        
        return features
    
    def correct_surname(self, misspelled: str, top_k: int = 5) -> Tuple[str, float]:
        """
        Correct a potentially misspelled surname.
        
        Args:
            misspelled: Potentially misspelled surname
            top_k: Number of candidates to consider
            
        Returns:
            Tuple of (corrected surname, confidence score)
        """
        from src.feature_engineering import get_top_candidates, extract_features
        
        # Si la palabra es muy corta, devolverla sin cambios
        if not misspelled or len(misspelled) < 3:
            return misspelled, 0.0
        
        # Get top candidates
        candidates = get_top_candidates(misspelled, self.all_surnames, k=top_k)
        
        # If no candidates found, return the original
        if not candidates:
            return misspelled, 0.0
        
        # If the misspelled surname is in the reference list, return it
        if misspelled in self.all_surnames:
            return misspelled, 1.0
        
        # Extract features
        features_list = extract_features(misspelled, candidates)
        X = np.array([list(features.values()) for features in features_list])
        
        # Get prediction probabilities
        probs = self.correction_model.predict_proba(X)
        
        # Find the candidate with highest probability of being correct
        if probs.shape[1] > 1:  # Binary classification
            positive_probs = probs[:, 1]
            best_idx = np.argmax(positive_probs)
            confidence = positive_probs[best_idx]
        else:  # In case model returns only one probability
            best_idx = 0
            confidence = probs[0]
        
        # Return the best candidate and confidence
        return candidates[best_idx], float(confidence)
    
    def correct_text(self, text: str, confidence_threshold: float = 0.7, 
                    surname_confidence_threshold: float = 0.6) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Correct all potential surnames in a text using improved detection.
        
        Args:
            text: Text containing potentially misspelled surnames
            confidence_threshold: Minimum confidence for correction
            surname_confidence_threshold: Minimum confidence for surname detection
            
        Returns:
            Tuple of (corrected text, list of corrections)
        """
        import re
        from src.surname_detection import extract_potential_surnames
        
        # Extract potential surnames with context using the improved method
        potential_surnames_with_context = extract_potential_surnames(
            text, 
            use_ner=self.use_ner,
            common_places=self.common_places,
            common_words=self.common_words
        )
        
        # Track corrections
        corrections = []
        corrected_text = text
        
        # Process each potential surname
        for surname, context in potential_surnames_with_context:
            # Skip very short words
            if len(surname) < 3:
                continue
            
            # Verificar si realmente es un apellido
            is_surname_result, surname_confidence = self.is_surname(surname, context)
            
            # Solo procesar si es probablemente un apellido
            if is_surname_result and surname_confidence >= surname_confidence_threshold:
                # Correct the surname
                corrected, correction_confidence = self.correct_surname(surname)
                
                # Apply correction if confidence is high enough and it's actually different
                if correction_confidence >= confidence_threshold and corrected.lower() != surname.lower():
                    # Replace the surname in the text (preserving case)
                    pattern = r'\b' + re.escape(surname) + r'\b'
                    corrected_text = re.sub(pattern, corrected, corrected_text)
                    
                    # Track the correction
                    corrections.append({
                        'original': surname,
                        'corrected': corrected,
                        'correction_confidence': correction_confidence,
                        'surname_confidence': surname_confidence,
                        'context': context
                    })
        
        return corrected_text, corrections
    
    def save(self, model_path: str) -> None:
        """
        Save the model and reference surnames.
        
        Args:
            model_path: Path to save the model
        """
        # Save the model
        self.model.save(model_path)
        
        # Save the reference surnames
        surnames_path = os.path.join(os.path.dirname(model_path), 'reference_surnames.pkl')
        with open(surnames_path, 'wb') as f:
            pickle.dump(self.all_surnames, f)
    
    def load(self, model_path: str) -> None:
        """
        Load the model and reference surnames.
        
        Args:
            model_path: Path to the saved model
        """
        # Load the model
        self.model.load(model_path)
        
        # Load the reference surnames
        surnames_path = os.path.join(os.path.dirname(model_path), 'reference_surnames.pkl')
        if os.path.exists(surnames_path):
            with open(surnames_path, 'rb') as f:
                self.all_surnames = pickle.load(f)


def load_model(model_path: str) -> Any:
    """
    Load a machine learning model from a file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
    """
    print(f"Cargando modelo desde {model_path}...")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Verificar el tipo de modelo
        model_type = type(model).__name__
        print(f"Modelo cargado correctamente: {model_type}")
        
        # Si es un modelo de scikit-learn, devolverlo directamente
        if model_type in ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier']:
            return model
        # Si es uno de nuestros modelos personalizados, devolverlo directamente
        elif hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise


def create_ensemble_model(models: List[BaseModel], weights: Optional[List[float]] = None) -> callable:
    """
    Create an ensemble model from multiple base models.
    
    Args:
        models: List of trained models
        weights: Optional weights for each model (default: equal weights)
        
    Returns:
        Function that takes features and returns predictions
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    def ensemble_predict(X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        # Get predictions from each model
        all_probs = []
        for model, weight in zip(models, weights):
            probs = model.predict_proba(X)
            all_probs.append(probs * weight)
        
        # Combine predictions
        combined_probs = sum(all_probs)
        
        # Return class with highest probability
        return np.argmax(combined_probs, axis=1)
    
    return ensemble_predict

"""
Feature engineering module for surname correction project.
Handles creating features for machine learning models and generating synthetic data.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any
from fuzzywuzzy import fuzz
import re


def generate_misspellings(word: str, num: int = 5) -> List[str]:
    """
    Generate synthetic misspellings of a word for data augmentation.
    
    Args:
        word: Original word to generate misspellings for
        num: Number of misspellings to generate
        
    Returns:
        List of misspelled versions of the word
    """
    if not word or len(word) < 3:
        return [word]
    
    misspellings = []
    operations = ['substitute', 'delete', 'insert', 'swap', 'repeat', 'case']
    
    for _ in range(num):
        misspelled = word
        # Apply 1-2 random operations
        for _ in range(random.randint(1, 2)):
            operation = random.choice(operations)
            
            if operation == 'substitute' and len(misspelled) > 0:
                # Substitute a random character
                pos = random.randint(0, len(misspelled) - 1)
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                misspelled = misspelled[:pos] + char + misspelled[pos+1:]
                
            elif operation == 'delete' and len(misspelled) > 3:
                # Delete a random character
                pos = random.randint(0, len(misspelled) - 1)
                misspelled = misspelled[:pos] + misspelled[pos+1:]
                
            elif operation == 'insert':
                # Insert a random character
                pos = random.randint(0, len(misspelled))
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                misspelled = misspelled[:pos] + char + misspelled[pos:]
                
            elif operation == 'swap' and len(misspelled) > 1:
                # Swap two adjacent characters
                pos = random.randint(0, len(misspelled) - 2)
                misspelled = misspelled[:pos] + misspelled[pos+1] + misspelled[pos] + misspelled[pos+2:]
                
            elif operation == 'repeat' and len(misspelled) > 0:
                # Repeat a character
                pos = random.randint(0, len(misspelled) - 1)
                misspelled = misspelled[:pos] + misspelled[pos] + misspelled[pos:]
                
            elif operation == 'case' and len(misspelled) > 0:
                # Change case of a random character
                pos = random.randint(0, len(misspelled) - 1)
                if misspelled[pos].islower():
                    misspelled = misspelled[:pos] + misspelled[pos].upper() + misspelled[pos+1:]
                else:
                    misspelled = misspelled[:pos] + misspelled[pos].lower() + misspelled[pos+1:]
        
        if misspelled != word and misspelled not in misspellings:
            misspellings.append(misspelled)
    
    # If we couldn't generate enough unique misspellings, fill with what we have
    while len(misspellings) < num:
        if misspellings:
            misspellings.append(misspellings[0])
        else:
            misspellings.append(word)
    
    return misspellings


def extract_features(misspelled: str, candidates: List[str]) -> List[Dict[str, float]]:
    """
    Extract features for machine learning models based on string similarity metrics.
    
    Args:
        misspelled: Potentially misspelled surname
        candidates: List of candidate correct surnames
        
    Returns:
        List of feature dictionaries for each candidate
    """
    features_list = []
    
    for candidate in candidates:
        features = {}
        
        # String similarity features
        features['length_diff'] = abs(len(misspelled) - len(candidate)) / max(len(misspelled), len(candidate))
        features['levenshtein_ratio'] = fuzz.ratio(misspelled.lower(), candidate.lower()) / 100.0
        features['partial_ratio'] = fuzz.partial_ratio(misspelled.lower(), candidate.lower()) / 100.0
        features['token_sort_ratio'] = fuzz.token_sort_ratio(misspelled.lower(), candidate.lower()) / 100.0
        
        # Character-level features
        features['first_char_match'] = 1.0 if misspelled and candidate and misspelled[0].lower() == candidate[0].lower() else 0.0
        features['last_char_match'] = 1.0 if misspelled and candidate and misspelled[-1].lower() == candidate[-1].lower() else 0.0
        
        # Phonetic features
        features['soundex_match'] = get_soundex_similarity(misspelled, candidate)
        
        # N-gram features
        features['bigram_similarity'] = get_ngram_similarity(misspelled, candidate, n=2)
        features['trigram_similarity'] = get_ngram_similarity(misspelled, candidate, n=3)
        
        features_list.append(features)
    
    return features_list


def get_soundex_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity based on Soundex phonetic algorithm.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score (0-1)
    """
    from jellyfish import soundex
    
    try:
        soundex1 = soundex(s1)
        soundex2 = soundex(s2)
        
        # Calculate similarity based on Soundex codes
        if soundex1 == soundex2:
            return 1.0
        
        # Count matching positions
        matches = sum(1 for a, b in zip(soundex1, soundex2) if a == b)
        return matches / max(len(soundex1), len(soundex2))
    except:
        # Fallback if jellyfish is not available
        return 0.0 if s1.lower() != s2.lower() else 1.0


def get_ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """
    Calculate similarity based on character n-grams.
    
    Args:
        s1: First string
        s2: Second string
        n: Size of n-grams
        
    Returns:
        Similarity score (0-1)
    """
    # Convert to lowercase
    s1, s2 = s1.lower(), s2.lower()
    
    # Generate n-grams
    ngrams1 = [s1[i:i+n] for i in range(len(s1) - n + 1)]
    ngrams2 = [s2[i:i+n] for i in range(len(s2) - n + 1)]
    
    # Empty strings case
    if not ngrams1 or not ngrams2:
        return 0.0 if s1 != s2 else 1.0
    
    # Count common n-grams
    common = set(ngrams1) & set(ngrams2)
    
    # Calculate Jaccard similarity
    return len(common) / (len(set(ngrams1) | set(ngrams2)))


def create_feature_matrix(misspelled_surnames: List[str], 
                          correct_surnames: List[str], 
                          all_correct_surnames: List[str],
                          top_k: int = 5) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature matrix for training machine learning models.
    
    Args:
        misspelled_surnames: List of misspelled surnames
        correct_surnames: List of corresponding correct surnames
        all_correct_surnames: Complete list of all correct surnames
        top_k: Number of candidates to consider for each misspelled surname
        
    Returns:
        Tuple of (feature matrix, target labels)
    """
    from sklearn.preprocessing import StandardScaler
    
    X = []
    y = []
    
    for misspelled, correct in zip(misspelled_surnames, correct_surnames):
        # Find top-k candidates based on Levenshtein distance
        candidates = get_top_candidates(misspelled, all_correct_surnames, k=top_k)
        
        # Extract features for each candidate
        features_list = extract_features(misspelled, candidates)
        
        # Add features to X
        for i, features in enumerate(features_list):
            X.append(list(features.values()))
            
            # Label is 1 if this candidate is the correct surname, 0 otherwise
            is_correct = candidates[i].lower() == correct.lower()
            y.append(1 if is_correct else 0)
    
    # Convert to numpy array
    X = np.array(X)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def get_top_candidates(misspelled: str, 
                       all_surnames: List[str], 
                       k: int = 5) -> List[str]:
    """
    Get top-k candidate surnames based on string similarity.
    
    Args:
        misspelled: Misspelled surname
        all_surnames: List of all correct surnames
        k: Number of candidates to return
        
    Returns:
        List of top-k candidate surnames
    """
    # Calculate similarity scores
    scores = [(surname, fuzz.ratio(misspelled.lower(), surname.lower())) 
              for surname in all_surnames]
    
    # Sort by score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k candidates
    return [surname for surname, _ in scores[:k]]


def prepare_embedding_features(texts: List[str]) -> np.ndarray:
    """
    Prepare text embedding features using pre-trained models.
    
    Args:
        texts: List of text strings
        
    Returns:
        Array of embedding vectors
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load pre-trained model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Generate embeddings
        embeddings = model.encode(texts)
        
        return embeddings
    except ImportError:
        # Fallback to simpler approach if sentence_transformers is not available
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        return vectorizer.fit_transform(texts).toarray()

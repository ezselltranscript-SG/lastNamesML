"""
Script to train the surname correction model.
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import pickle
import time

from src.data_processing import (
    load_reference_data,
    extract_names_surnames,
    load_transcriptions,
    create_training_data,
    split_data
)
from src.feature_engineering import (
    create_feature_matrix,
    generate_misspellings
)
from src.models import (
    RandomForestModel,
    SVMModel,
    NeuralNetworkModel,
    SurnameCorrector
)
from src.evaluation import (
    evaluate_model,
    cross_validate,
    plot_metrics,
    generate_evaluation_report
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train surname correction model')
    
    parser.add_argument('--reference-data', type=str, required=True,
                        help='Path to Excel file with reference names and surnames')
    parser.add_argument('--transcriptions-dir', type=str, default='data/transcriptions',
                        help='Directory containing transcription files')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'svm', 'neural_network', 'ensemble'],
                        help='Type of model to train')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save trained model')
    parser.add_argument('--name-col', type=str, default='full_name',
                        help='Column name in Excel file containing full names')
    parser.add_argument('--augment', action='store_true',
                        help='Augment training data with synthetic misspellings')
    parser.add_argument('--cross-validate', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    
    return parser.parse_args()


def main():
    """Main function to train the model."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading reference data from {args.reference_data}")
    # Load reference data
    reference_df = load_reference_data(args.reference_data)
    if reference_df.empty:
        print("Error: Reference data is empty")
        return
    
    # Extract names and surnames
    print(f"Extracting names and surnames from column '{args.name_col}'")
    try:
        names, surnames = extract_names_surnames(reference_df, args.name_col)
        print(f"Extracted {len(names)} names and {len(surnames)} surnames")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load transcriptions
    print(f"Loading transcriptions from {args.transcriptions_dir}")
    transcriptions = load_transcriptions(args.transcriptions_dir)
    print(f"Loaded {len(transcriptions)} transcription files")
    
    # Create training data
    print("Creating training data")
    if not transcriptions:
        print("No transcriptions found. Using synthetic misspellings only.")
        
    misspelled_surnames, correct_surnames = create_training_data(
        surnames, transcriptions, augment=args.augment
    )
    print(f"Created {len(misspelled_surnames)} training examples")
    
    # If we don't have enough training data, generate more synthetic misspellings
    if len(misspelled_surnames) < 100:
        print("Not enough training data. Generating additional synthetic misspellings.")
        for surname in surnames[:min(100, len(surnames))]:
            synthetic_misspellings = generate_misspellings(surname, num=5)
            misspelled_surnames.extend(synthetic_misspellings)
            correct_surnames.extend([surname] * len(synthetic_misspellings))
        print(f"Expanded to {len(misspelled_surnames)} training examples")
    
    # Create feature matrix
    print("Creating feature matrix")
    X, y = create_feature_matrix(misspelled_surnames, correct_surnames, surnames)
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
    print(f"Training set: {X_train.shape[0]} examples")
    print(f"Test set: {X_test.shape[0]} examples")
    
    # Train model
    print(f"Training {args.model_type} model")
    start_time = time.time()
    
    if args.model_type == 'random_forest':
        model = RandomForestModel()
    elif args.model_type == 'svm':
        model = SVMModel()
    elif args.model_type == 'neural_network':
        model = NeuralNetworkModel()
    elif args.model_type == 'ensemble':
        # Train multiple models for ensemble
        rf_model = RandomForestModel()
        svm_model = SVMModel()
        nn_model = NeuralNetworkModel()
        
        print("Training Random Forest model")
        rf_model.train(X_train, y_train)
        
        print("Training SVM model")
        svm_model.train(X_train, y_train)
        
        print("Training Neural Network model")
        nn_model.train(X_train, y_train)
        
        # Create ensemble
        from src.models import create_ensemble_model
        ensemble_predict = create_ensemble_model([rf_model, svm_model, nn_model])
        
        # Evaluate ensemble
        y_pred = ensemble_predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        print("Ensemble model metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save individual models
        rf_model.save(os.path.join(args.output_dir, 'random_forest_model.pkl'))
        svm_model.save(os.path.join(args.output_dir, 'svm_model.pkl'))
        nn_model.save(os.path.join(args.output_dir, 'neural_network_model.pkl'))
        
        # Use Random Forest as the base model for the corrector
        model = rf_model
    
    # Train the model (if not ensemble)
    if args.model_type != 'ensemble':
        model.train(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model")
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print("Model metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    if args.cross_validate:
        print("Performing cross-validation")
        if args.model_type == 'random_forest':
            cv_metrics = cross_validate(RandomForestModel, X, y)
        elif args.model_type == 'svm':
            cv_metrics = cross_validate(SVMModel, X, y)
        elif args.model_type == 'neural_network':
            cv_metrics = cross_validate(NeuralNetworkModel, X, y)
        else:
            print("Cross-validation not supported for ensemble model")
            cv_metrics = {}
        
        if cv_metrics:
            print("Cross-validation metrics:")
            for metric, value in cv_metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # Create surname corrector
    corrector = SurnameCorrector(model_type=args.model_type)
    corrector.model = model
    corrector.set_reference_surnames(surnames)
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{args.model_type}_model.pkl')
    print(f"Saving model to {model_path}")
    corrector.save(model_path)
    
    # Save reference data
    reference_path = os.path.join(args.output_dir, 'reference_data.pkl')
    with open(reference_path, 'wb') as f:
        pickle.dump({'names': names, 'surnames': surnames}, f)
    
    # Generate evaluation report
    report = generate_evaluation_report(
        metrics, args.model_type, 
        os.path.join(args.output_dir, 'evaluation_report.txt')
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script to generate training dataset for Job Placement Predictor.
This creates a synthetic dataset based on company hiring patterns.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from model.data_processor import DataProcessor

def main():
    print("\n" + "="*60)
    print("Job Placement Predictor - Training Data Generator")
    print("="*60 + "\n")

    processor = DataProcessor()
    
    # Generate training dataset
    print("Generating training dataset...\n")
    output_path = os.path.join(os.path.dirname(__file__), 'training_dataset.csv')
    
    try:
        processor.create_company_training_data(output_path)
        print("\nTraining dataset generated successfully!")
        print(f"  Location: {output_path}")
        print("\nYou can now train the model using this dataset.")
    except Exception as e:
        print(f"\nError generating training data: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

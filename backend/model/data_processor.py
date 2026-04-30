import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os

class DataProcessor:
    """
    Processes and prepares data for the ML model.
    Handles data cleaning, feature engineering, and transformation.
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.column_mappings = {}
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        """
        try:
            self.data = pd.read_csv(csv_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean data by handling missing values and duplicates.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        cleaned_data = self.data.copy()
        
        # Remove duplicates
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        print(f"Removed {initial_rows - len(cleaned_data)} duplicate rows")
        
        # Handle missing values
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().sum() > 0:
                if cleaned_data[col].dtype in ['float64', 'int64']:
                    cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
                else:
                    cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
        
        print(f"Data cleaned. Final shape: {cleaned_data.shape}")
        return cleaned_data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical features.
        """
        from sklearn.preprocessing import LabelEncoder
        
        encoded_data = data.copy()
        encoders = {}
        
        categorical_columns = encoded_data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col])
            encoders[col] = le
        
        return encoded_data, encoders
    
    def normalize_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        """
        Normalize numerical features to 0-1 range.
        """
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
        
        normalized_data = data.copy()
        normalized_data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        
        return normalized_data, scaler
    
    def create_company_training_data(self, output_path: str) -> None:
        """
        Create sample training data based on company hiring patterns.
        This generates synthetic data for initial model training.
        """
        np.random.seed(42)
        
        n_samples = 500
        
        # Company hiring requirements and patterns
        company_profiles = {
            'FAANG': {
                'min_cgpa': 7.5,
                'avg_skills': 6,
                'avg_projects': 4,
                'placement_rate': 0.95
            },
            'Product Companies': {
                'min_cgpa': 7.0,
                'avg_skills': 5,
                'avg_projects': 3,
                'placement_rate': 0.85
            },
            'Startups': {
                'min_cgpa': 6.5,
                'avg_skills': 4,
                'avg_projects': 3,
                'placement_rate': 0.75
            },
            'Service Companies': {
                'min_cgpa': 6.0,
                'avg_skills': 3,
                'avg_projects': 1,
                'placement_rate': 0.65
            }
        }
        
        skills_list = [
            'Python', 'Java', 'JavaScript', 'C++', 'Go',
            'Machine Learning', 'Web Development', 'Cloud Computing',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'React'
        ]
        
        certifications_list = [
            'AWS Solutions', 'Azure Admin', 'Google Cloud',
            'Kubernetes Admin', 'Security+', 'JIRA', 'None'
        ]
        
        domains_list = [
            'Web Development', 'Mobile Development', 'Data Science',
            'Cloud Computing', 'DevOps', 'Machine Learning'
        ]
        
        data = []
        
        for _ in range(n_samples):
            company_type = np.random.choice(list(company_profiles.keys()))
            profile = company_profiles[company_type]
            
            cgpa = np.random.normal(
                profile['min_cgpa'] + 1,
                0.8
            )
            cgpa = np.clip(cgpa, 4.0, 10.0)
            
            num_skills = np.random.poisson(profile['avg_skills']) + 1
            skills_string = ', '.join(np.random.choice(skills_list, size=min(num_skills, len(skills_list)), replace=False))
            
            num_certifications = np.random.randint(0, 3)
            certifications = ', '.join(np.random.choice(certifications_list, size=num_certifications))
            
            projects = np.random.poisson(profile['avg_projects'])
            domain = np.random.choice(domains_list)
            internships = np.random.randint(0, 3)
            communication_level = np.random.randint(1, 6)
            
            # Calculate placement probability based on features
            placement_prob = (
                (cgpa / 10) * 0.3 +
                (min(num_skills, 6) / 6) * 0.25 +
                (min(num_certifications, 2) / 2) * 0.15 +
                (min(projects, 5) / 5) * 0.15 +
                (min(internships, 2) / 2) * 0.1 +
                (communication_level / 5) * 0.05
            )
            
            placed = 1 if np.random.random() < placement_prob else 0
            
            data.append({
                'cgpa': round(cgpa, 2),
                'skills': skills_string,
                'certifications': certifications,
                'projects': projects,
                'domain': domain,
                'internships': internships,
                'communication_level': communication_level,
                'company_type': company_type,
                'placed': placed
            })
        
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Training data created successfully at {output_path}")
        print(f"Total samples: {len(df)}")
        print(f"Placement rate: {df['placed'].mean():.2%}")
        print(f"\nData preview:\n{df.head()}")
        
        return df

if __name__ == '__main__':
    # Example usage
    processor = DataProcessor()
    output_path = '../data/training_dataset.csv'
    processor.create_company_training_data(output_path)

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from typing import Dict, List, Any

class JobPlacementPredictor:
    """
    Machine Learning model using Naïve Bayes algorithm to predict job placement.
    
    Features:
    - CGPA: Grade Point Average (0-10)
    - Skills: List of technical skills
    - Certifications: Professional certifications
    - Projects: Number of completed projects
    - Domain: Career domain interest
    - Internships: Number of internships
    - Communication Level: Soft skills rating (1-5)
    """
    
    def __init__(self):
        self.model = GaussianNB()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.metrics = {}
        
        # Define domain-based skill requirements
        self.domain_skills = {
            'Web Development': ['JavaScript', 'React', 'HTML', 'CSS', 'Node.js', 'Django'],
            'Mobile Development': ['Java', 'Swift', 'React Native', 'Kotlin', 'Flutter'],
            'Data Science': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'R'],
            'Machine Learning': ['Python', 'TensorFlow', 'PyTorch', 'Deep Learning', 'NLP'],
            'Cloud Computing': ['AWS', 'Azure', 'Docker', 'Kubernetes', 'GCP'],
            'DevOps': ['Docker', 'Kubernetes', 'CI/CD', 'Jenkins', 'AWS', 'Linux'],
            'Cybersecurity': ['Linux', 'Networking', 'Security+', 'Ethical Hacking', 'Firewalls'],
            'Database Administration': ['SQL', 'PostgreSQL', 'MongoDB', 'Oracle', 'NoSQL'],
            'Software Development': ['C++', 'Java', 'Design Patterns', 'Git', 'Testing'],
            'AI/NLP': ['Python', 'NLP', 'Machine Learning', 'Deep Learning', 'TensorFlow'],
            'Full Stack Development': ['JavaScript', 'React', 'Node.js', 'MongoDB', 'Docker'],
            'Embedded Systems': ['C', 'C++', 'IoT', 'ARM', 'RTOS'],
            'Game Development': ['C#', 'Unity', 'Unreal Engine', 'Game Physics']
        }
        
        # High-demand skills that increase placement probability
        self.high_demand_skills = [
            'Python', 'JavaScript', 'Machine Learning', 'AWS', 'Docker',
            'React', 'Java', 'Kubernetes', 'Azure', 'Cloud Computing'
        ]
    
    def _encode_features(self, data: pd.DataFrame, fit=False) -> np.ndarray:
        """
        Encode categorical features using LabelEncoder.
        """
        encoded_data = data.copy()
        
        # Encode categorical columns
        categorical_columns = encoded_data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                if fit:
                    encoded_data[col] = self.label_encoders[col].fit_transform(encoded_data[col])
            else:
                encoded_data[col] = self.label_encoders[col].transform(encoded_data[col])
        
        return encoded_data.values
    
    def _calculate_skill_match(self, user_skills: List[str], domain: str) -> float:
        """
        Calculate how well user's skills match the domain requirements.
        """
        required_skills = self.domain_skills.get(domain, [])
        if not required_skills:
            return 0.5  # Neutral score if domain not found
        
        matches = sum(1 for skill in user_skills if skill in required_skills)
        return min(matches / len(required_skills), 1.0)
    
    def _calculate_skill_quality(self, skills: List[str]) -> float:
        """
        Calculate quality score based on high-demand skills.
        """
        high_demand_count = sum(1 for skill in skills if skill in self.high_demand_skills)
        total_skills = len(skills) if skills else 1
        return min(high_demand_count / total_skills, 1.0)
    
    def train(self, csv_path: str) -> Dict[str, float]:
        """
        Train the Naïve Bayes model on the provided dataset.
        
        Args:
            csv_path: Path to the training CSV file
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Prepare features and target
            X = df.drop('placed', axis=1, errors='ignore')
            y = df['placed'] if 'placed' in df.columns else df.iloc[:, -1]
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Encode features
            X_encoded = self._encode_features(X, fit=True)
            X_scaled = self.scaler.fit_transform(X_encoded)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Calculate metrics
            y_pred = self.model.predict(X_test)
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }
            
            print(f"Model trained successfully!")
            print(f"Accuracy: {self.metrics['accuracy']:.4f}")
            print(f"Precision: {self.metrics['precision']:.4f}")
            print(f"Recall: {self.metrics['recall']:.4f}")
            print(f"F1-Score: {self.metrics['f1_score']:.4f}")
            
            return self.metrics
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
    
    def predict(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict job placement probability for a candidate.
        
        Args:
            candidate_data: Dictionary containing candidate features
                - cgpa: float (0-10)
                - skills: list of strings
                - certifications: list of strings
                - projects: int
                - domain: string
                - internships: int
                - communication_level: int (1-5)
                
        Returns:
            Dictionary containing prediction results
        """
        try:
            if not self.is_trained:
                return self._get_heuristic_prediction(candidate_data)
            
            # Extract features
            cgpa = candidate_data.get('cgpa', 0)
            skills = candidate_data.get('skills', [])
            certifications = candidate_data.get('certifications', [])
            projects = candidate_data.get('projects', 0)
            domain = candidate_data.get('domain', 'Software Development')
            internships = candidate_data.get('internships', 0)
            communication_level = candidate_data.get('communication_level', 3)
            
            # Calculate derived features
            skill_match_score = self._calculate_skill_match(skills, domain)
            skill_quality_score = self._calculate_skill_quality(skills)
            certification_count = len(certifications)
            skill_count = len(skills)
            
            # Create feature vector
            features = np.array([
                cgpa,
                skill_count,
                certification_count,
                projects,
                internships,
                communication_level,
                skill_match_score * 10,  # Scale to 0-10
                skill_quality_score * 10  # Scale to 0-10
            ]).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probability
            probabilities = self.model.predict_proba(features_scaled)[0]
            placement_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            
            # Calculate confidence based on feature strength
            confidence = min(
                (cgpa / 10) * 0.3 +
                (skill_match_score) * 0.3 +
                (certification_count / 3) * 0.2 +
                (min(projects, 5) / 5) * 0.1 +
                (min(internships, 2) / 2) * 0.1,
                1.0
            )
            
            # Determine status
            if placement_probability >= 0.7:
                status = "Likely to be Placed ✓"
            elif placement_probability >= 0.5:
                status = "Moderate Chances"
            else:
                status = "Low Chances - Improvement Needed"
            
            # Generate recommendations
            recommendation = self._generate_recommendation(candidate_data, placement_probability)
            
            return {
                'probability': placement_probability,
                'confidence': confidence,
                'status': status,
                'recommendation': recommendation
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return self._get_heuristic_prediction(candidate_data)
    
    def _get_heuristic_prediction(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction using heuristics when model is not trained.
        """
        cgpa = candidate_data.get('cgpa', 0)
        skills = candidate_data.get('skills', [])
        certifications = candidate_data.get('certifications', [])
        projects = candidate_data.get('projects', 0)
        internships = candidate_data.get('internships', 0)
        communication_level = candidate_data.get('communication_level', 3)
        domain = candidate_data.get('domain', 'Software Development')
        
        # Calculate probability using weighted heuristics
        cgpa_score = (cgpa / 10) * 0.3
        skills_score = min(len(skills) / 5, 1.0) * 0.25
        certification_score = min(len(certifications) / 3, 1.0) * 0.15
        project_score = min(projects / 5, 1.0) * 0.15
        internship_score = min(internships / 3, 1.0) * 0.1
        communication_score = (communication_level / 5) * 0.05
        
        skill_match = self._calculate_skill_match(skills, domain) * 0.1
        
        probability = min(
            cgpa_score + skills_score + certification_score + 
            project_score + internship_score + communication_score + skill_match,
            1.0
        )
        
        confidence = min(
            (cgpa / 10) * 0.3 +
            (len(skills) / 5) * 0.3 +
            (len(certifications) / 3) * 0.2 +
            (min(projects, 5) / 5) * 0.1 +
            (min(internships, 2) / 2) * 0.1,
            1.0
        )
        
        if probability >= 0.7:
            status = "Likely to be Placed ✓"
        elif probability >= 0.5:
            status = "Moderate Chances"
        else:
            status = "Low Chances - Improvement Needed"
        
        recommendation = self._generate_recommendation(candidate_data, probability)
        
        return {
            'probability': probability,
            'confidence': confidence,
            'status': status,
            'recommendation': recommendation
        }
    
    def _generate_recommendation(self, candidate_data: Dict[str, Any], probability: float) -> str:
        """
        Generate personalized recommendations based on candidate profile.
        """
        recommendations = []
        
        cgpa = candidate_data.get('cgpa', 0)
        skills = candidate_data.get('skills', [])
        certifications = candidate_data.get('certifications', [])
        projects = candidate_data.get('projects', 0)
        internships = candidate_data.get('internships', 0)
        communication_level = candidate_data.get('communication_level', 3)
        domain = candidate_data.get('domain', 'Software Development')
        
        # CGPA feedback
        if cgpa < 6:
            recommendations.append("Focus on improving academic performance")
        elif cgpa < 7:
            recommendations.append("Maintain consistent academic performance")
        
        # Skills feedback
        required_skills = self.domain_skills.get(domain, [])
        missing_skills = [s for s in required_skills if s not in skills]
        if missing_skills:
            recommendations.append(f"Learn key skills: {', '.join(missing_skills[:2])}")
        
        # Projects feedback
        if projects < 2:
            recommendations.append("Complete more real-world projects")
        
        # Internship feedback
        if internships == 0:
            recommendations.append("Gain internship experience")
        
        # Communication feedback
        if communication_level < 3:
            recommendations.append("Improve communication and interpersonal skills")
        
        # Certification feedback
        if len(certifications) == 0:
            recommendations.append("Pursue relevant industry certifications")
        
        # Overall feedback
        if probability >= 0.7:
            recommendations.insert(0, "Strong profile - Focus on interview preparation")
        elif probability < 0.5:
            recommendations.insert(0, "Develop multiple skill areas for better placement")
        else:
            recommendations.insert(0, "Continue skill development and gain more experience")
        
        return " | ".join(recommendations[:3])  # Return top 3 recommendations
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_data = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'metrics': self.metrics
            }
            joblib.dump(model_data, path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        """
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            self.metrics = model_data['metrics']
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

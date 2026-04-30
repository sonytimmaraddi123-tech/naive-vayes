from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.naive_bayes_model import JobPlacementPredictor

app = Flask(__name__)
CORS(app)

# Initialize the predictor
predictor = None

def init_predictor():
    global predictor
    try:
        predictor = JobPlacementPredictor()
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'naive_bayes_model.pkl')
        
        if os.path.exists(model_path):
            predictor.load_model(model_path)
        else:
            # Train on default dataset
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
            if os.path.exists(data_path):
                predictor.train(data_path)
                # Create models directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                predictor.save_model(model_path)
            else:
                print(f"Warning: Training data not found at {data_path}")
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")

@app.before_request
def before_request():
    global predictor
    if predictor is None:
        init_predictor()

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict job placement probability
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['cgpa', 'skills', 'certifications', 'projects', 'domain', 'internships', 'communication_level']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Extract data
        cgpa = float(data['cgpa'])
        skills = data['skills'] if isinstance(data['skills'], list) else [data['skills']]
        certifications = data['certifications'] if isinstance(data['certifications'], list) else [data['certifications']]
        projects = int(data['projects'])
        domain = data['domain']
        internships = int(data['internships'])
        communication_level = int(data['communication_level'])
        
        # Validate ranges
        if not (0 <= cgpa <= 10):
            return jsonify({'error': 'CGPA must be between 0 and 10', 'status': 'error'}), 400
        if not (1 <= communication_level <= 5):
            return jsonify({'error': 'Communication level must be between 1 and 5', 'status': 'error'}), 400
        if projects < 0:
            return jsonify({'error': 'Projects cannot be negative', 'status': 'error'}), 400
        if internships < 0:
            return jsonify({'error': 'Internships cannot be negative', 'status': 'error'}), 400
        
        # Make prediction
        result = predictor.predict({
            'cgpa': cgpa,
            'skills': skills,
            'certifications': certifications,
            'projects': projects,
            'domain': domain,
            'internships': internships,
            'communication_level': communication_level
        })
        
        return jsonify({
            'placement_probability': round(result['probability'], 3),
            'confidence': round(result['confidence'], 3),
            'status': result['status'],
            'recommendation': result['recommendation'],
            'success': True
        }), 200
        
    except ValueError as ve:
        return jsonify({'error': f'Invalid data format: {str(ve)}', 'status': 'error'}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/skills', methods=['GET'])
def get_skills():
    """
    API endpoint to get available skills
    """
    skills = {
        'technical': [
            'Python', 'Java', 'C++', 'JavaScript', 'C#', 'Go', 'Rust',
            'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask',
            'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL',
            'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
            'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes',
            'Git', 'CI/CD', 'Linux', 'Windows'
        ],
        'soft_skills': [
            'Leadership', 'Team Collaboration', 'Problem Solving',
            'Communication', 'Critical Thinking', 'Time Management'
        ]
    }
    return jsonify(skills), 200

@app.route('/api/certifications', methods=['GET'])
def get_certifications():
    """
    API endpoint to get popular certifications
    """
    certifications = [
        'AWS Solutions Architect',
        'AWS Developer Associate',
        'Azure Administrator',
        'Azure Solutions Architect',
        'Google Cloud Associate',
        'Kubernetes Administrator',
        'Certified Data Scientist',
        'TensorFlow Developer',
        'Oracle Java Programmer',
        'CompTIA Security+',
        'Certified Ethical Hacker',
        'Scrum Master'
    ]
    return jsonify(certifications), 200

@app.route('/api/domains', methods=['GET'])
def get_domains():
    """
    API endpoint to get career domains
    """
    domains = [
        'Web Development',
        'Mobile Development',
        'Data Science',
        'Machine Learning',
        'Cloud Computing',
        'DevOps',
        'Cybersecurity',
        'Database Administration',
        'Software Development',
        'AI/NLP',
        'Full Stack Development',
        'Embedded Systems',
        'Game Development'
    ]
    return jsonify(domains), 200

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({'status': 'healthy', 'message': 'Server is running'}), 200

@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint
    """
    return jsonify({
        'name': 'Job Placement Predictor API',
        'version': '1.0.0',
        'description': 'Predicts job placement probability using Naïve Bayes ML algorithm'
    }), 200

if __name__ == '__main__':
    init_predictor()
    app.run(debug=True, host='0.0.0.0', port=5000)

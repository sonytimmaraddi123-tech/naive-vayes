// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const form = document.getElementById('predictionForm');
const resultsSection = document.getElementById('resultsSection');
const skillsInput = document.getElementById('skillsInput');
const skillsSuggestions = document.getElementById('skillsSuggestions');
const skillsTags = document.getElementById('skillsTags');
const certInput = document.getElementById('certInput');
const certSuggestions = document.getElementById('certSuggestions');
const certTags = document.getElementById('certTags');

// State
let availableSkills = [];
let availableCertifications = [];
let selectedSkills = [];
let selectedCertifications = [];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    loadSkills();
    loadCertifications();
    setupEventListeners();
});

// Load available skills
async function loadSkills() {
    try {
        const response = await fetch(`${API_BASE_URL}/skills`);
        const data = await response.json();
        availableSkills = [...data.technical, ...data.soft_skills];
    } catch (error) {
        console.error('Error loading skills:', error);
        availableSkills = [
            'Python', 'Java', 'JavaScript', 'C++', 'Go', 'Rust',
            'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask',
            'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL',
            'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
            'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'
        ];
    }
}

// Load available certifications
async function loadCertifications() {
    try {
        const response = await fetch(`${API_BASE_URL}/certifications`);
        const data = await response.json();
        availableCertifications = data;
    } catch (error) {
        console.error('Error loading certifications:', error);
        availableCertifications = [
            'AWS Solutions Architect',
            'AWS Developer Associate',
            'Azure Administrator',
            'Azure Solutions Architect',
            'Google Cloud Associate',
            'Kubernetes Administrator',
            'Certified Data Scientist',
            'TensorFlow Developer',
            'Oracle Java Programmer',
            'CompTIA Security+'
        ];
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Form submission
    form.addEventListener('submit', handleFormSubmit);

    // Skills input
    skillsInput.addEventListener('input', handleSkillsInput);
    skillsInput.addEventListener('blur', () => {
        setTimeout(() => {
            skillsSuggestions.classList.remove('active');
        }, 200);
    });

    // Certifications input
    certInput.addEventListener('input', handleCertInput);
    certInput.addEventListener('blur', () => {
        setTimeout(() => {
            certSuggestions.classList.remove('active');
        }, 200);
    });
}

// Handle skills input
function handleSkillsInput(e) {
    const value = e.target.value.toLowerCase();
    const filtered = availableSkills.filter(skill =>
        skill.toLowerCase().includes(value) &&
        !selectedSkills.includes(skill)
    );

    displaySuggestions(filtered, skillsSuggestions, 'addSkill');
}

// Handle certifications input
function handleCertInput(e) {
    const value = e.target.value.toLowerCase();
    const filtered = availableCertifications.filter(cert =>
        cert.toLowerCase().includes(value) &&
        !selectedCertifications.includes(cert)
    );

    displaySuggestions(filtered, certSuggestions, 'addCertification');
}

// Display suggestions
function displaySuggestions(items, container, action) {
    container.innerHTML = '';

    if (items.length === 0) {
        container.classList.remove('active');
        return;
    }

    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.textContent = item;
        div.onclick = () => {
            if (action === 'addSkill') {
                addSkill(item);
            } else if (action === 'addCertification') {
                addCertification(item);
            }
        };
        container.appendChild(div);
    });

    container.classList.add('active');
}

// Add skill
function addSkill(skill) {
    if (!selectedSkills.includes(skill)) {
        selectedSkills.push(skill);
        renderSkillTags();
        skillsInput.value = '';
        skillsSuggestions.classList.remove('active');
    }
}

// Add certification
function addCertification(cert) {
    if (!selectedCertifications.includes(cert)) {
        selectedCertifications.push(cert);
        renderCertTags();
        certInput.value = '';
        certSuggestions.classList.remove('active');
    }
}

// Remove skill
function removeSkill(skill) {
    selectedSkills = selectedSkills.filter(s => s !== skill);
    renderSkillTags();
}

// Remove certification
function removeCertification(cert) {
    selectedCertifications = selectedCertifications.filter(c => c !== cert);
    renderCertTags();
}

// Render skill tags
function renderSkillTags() {
    skillsTags.innerHTML = '';
    selectedSkills.forEach(skill => {
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `
            ${skill}
            <button type="button" onclick="removeSkill('${skill}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        skillsTags.appendChild(tag);
    });
}

// Render certification tags
function renderCertTags() {
    certTags.innerHTML = '';
    selectedCertifications.forEach(cert => {
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `
            ${cert}
            <button type="button" onclick="removeCertification('${cert}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        certTags.appendChild(tag);
    });
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();

    // Validate selections
    if (selectedSkills.length === 0) {
        alert('Please select at least one skill');
        return;
    }

    // Prepare form data
    const formData = {
        cgpa: parseFloat(document.getElementById('cgpa').value),
        skills: selectedSkills,
        certifications: selectedCertifications,
        projects: parseInt(document.getElementById('projects').value),
        domain: document.getElementById('domain').value,
        internships: parseInt(document.getElementById('internships').value),
        communication_level: parseInt(document.getElementById('communication').value)
    };

    // Show loading state
    const submitButton = form.querySelector('.btn-submit');
    const originalText = submitButton.innerHTML;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    submitButton.disabled = true;

    try {
        // Make prediction
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (response.ok) {
            displayResults(result, formData);
            // Scroll to results
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }, 300);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    } finally {
        // Restore button
        submitButton.innerHTML = originalText;
        submitButton.disabled = false;
    }
}

// Display results
function displayResults(prediction, formData) {
    // Show results section
    resultsSection.style.display = 'block';

    // Update probability
    const percentage = Math.round(prediction.placement_probability * 100);
    document.getElementById('probabilityPercentage').textContent = percentage + '%';
    document.getElementById('placementStatus').textContent = prediction.status;
    document.getElementById('confidenceScore').textContent = `Confidence: ${Math.round(prediction.confidence * 100)}%`;

    // Update progress ring
    const circumference = 2 * Math.PI * 95; // radius = 95
    const offset = circumference - (prediction.placement_probability * circumference);
    const progressRing = document.getElementById('progressRing');
    progressRing.style.strokeDashoffset = offset;

    // Update recommendation
    document.getElementById('recommendationText').textContent = prediction.recommendation;

    // Update stats
    document.getElementById('statCGPA').textContent = formData.cgpa.toFixed(2);
    document.getElementById('statSkills').textContent = formData.skills.length;
    document.getElementById('statProjects').textContent = formData.projects;
    document.getElementById('statInternships').textContent = formData.internships;

    // Change progress ring color based on probability
    const color = prediction.placement_probability >= 0.7 ? '#4CAF50' :
                  prediction.placement_probability >= 0.5 ? '#FF9800' : '#f44336';
    progressRing.style.stroke = color;
}

// Utility functions
function showError(message) {
    alert(message);
}

function showSuccess(message) {
    alert(message);
}

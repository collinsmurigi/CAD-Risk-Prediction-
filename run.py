from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
from datetime import datetime, timedelta
import os
import csv
from io import StringIO
import joblib

# Import the ML model
try:
    from models.cad_ml_model import CADMLModel
except ImportError:
    print("ML model module not found. Will use simple model only.")
    CADMLModel = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cad_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Make 'now' available to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now}

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='clinician')
    full_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class PatientAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    # Patient Identification
    patient_first_name = db.Column(db.String(100), nullable=False, default='Unknown')
    patient_last_name = db.Column(db.String(100), nullable=False, default='Patient')
    patient_mrn = db.Column(db.String(50))
    patient_dob = db.Column(db.Date)
    patient_phone = db.Column(db.String(20))
    
    # Basic demographics
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    
    # Clinical measurements
    trestbps = db.Column(db.Integer)  # Resting blood pressure
    chol = db.Column(db.Float)  # Total Cholesterol
    hdl_cholesterol = db.Column(db.Float)  # HDL Cholesterol
    fbs = db.Column(db.Boolean)  # Fasting blood sugar > 120 mg/dl
    thalach = db.Column(db.Integer)  # Maximum heart rate achieved
    
    # Additional fields
    smoking_status = db.Column(db.String(20))
    diabetes_status = db.Column(db.Boolean)
    family_history_cad = db.Column(db.Boolean)
    physical_activity_level = db.Column(db.String(20))
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    risk_category = db.Column(db.String(20))
    probability_score = db.Column(db.Float)
    model_used = db.Column(db.String(50), default='ml_model')
    
    @property
    def full_name(self):
        return f"{self.patient_first_name} {self.patient_last_name}".strip()
    
    def calculate_age_from_dob(self):
        """Calculate age from date of birth"""
        if self.patient_dob:
            today = datetime.now().date()
            age = today.year - self.patient_dob.year
            # Adjust if birthday hasn't occurred this year
            if today.month < self.patient_dob.month or (today.month == self.patient_dob.month and today.day < self.patient_dob.day):
                age -= 1
            return age
        return self.age

# Simple CAD Model (always available as fallback)
class SimpleCADModel:
    """Fallback simple CAD Risk Prediction Model"""
    def __init__(self):
        self.coefficients = {
            'age': 0.05,
            'total_cholesterol': 0.01,
            'hdl_cholesterol': -0.03,
            'systolic_bp': 0.02,
            'smoking': 0.3,
            'diabetes': 0.4,
            'family_history': 0.2
        }
        self.intercept = -8.0
    
    def predict(self, data):
        """Calculate risk score"""
        try:
            score = self.intercept
            
            # Add contributions from each factor
            score += data['age'] * self.coefficients['age']
            score += data['total_cholesterol'] * self.coefficients['total_cholesterol']
            score += data['hdl_cholesterol'] * self.coefficients['hdl_cholesterol']
            score += data['systolic_bp'] * self.coefficients['systolic_bp']
            
            if data.get('smoking_status') == 'current':
                score += self.coefficients['smoking']
            if data.get('diabetes_status', False):
                score += self.coefficients['diabetes']
            if data.get('family_history_cad', False):
                score += self.coefficients['family_history']
            
            # Convert to probability using logistic function
            probability = 1 / (1 + np.exp(-score))
            
            # Categorize risk
            if probability < 0.1:
                category = "Low"
            elif probability < 0.2:
                category = "Medium"
            else:
                category = "High"
            
            return {
                'probability': round(probability * 100, 2),
                'category': category
            }
        except Exception as e:
            print(f"Simple model prediction error: {e}")
            return {
                'probability': 5.0,
                'category': 'Low'
            }

# Initialize models
simple_model = SimpleCADModel()
ml_model = None

# Try to initialize ML model if available
if CADMLModel:
    try:
        ml_model = CADMLModel(model_type='random_forest')
        if ml_model.load_model('models/cad_model.pkl'):
            print("✓ ML Model loaded successfully!")
        else:
            print("! No trained model found. Using simple rule-based model.")
            ml_model = None
    except Exception as e:
        print(f"! Error loading ML model: {e}")
        ml_model = None

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    assessments = PatientAssessment.query.filter_by(user_id=current_user.id)\
        .order_by(PatientAssessment.created_at.desc())\
        .limit(10)\
        .all()
    
    return render_template('dashboard.html', 
                         user=current_user,
                         assessments=assessments)

@app.route('/assessment', methods=['GET', 'POST'])
@login_required
def assessment():
    if request.method == 'POST':
        try:
            # Parse date of birth and calculate age
            patient_dob = None
            age = 0
            
            if request.form.get('patient_dob'):
                patient_dob = datetime.strptime(request.form['patient_dob'], '%Y-%m-%d').date()
                # Calculate age from DOB
                today = datetime.now().date()
                age = today.year - patient_dob.year
                if today.month < patient_dob.month or (today.month == patient_dob.month and today.day < patient_dob.day):
                    age -= 1
            
            # Validate age
            if age < 18 or age > 120:
                flash('Patient must be between 18 and 120 years old', 'danger')
                return render_template('assessment.html')
            
            # Collect form data
            data = {
                # Patient Identification
                'patient_first_name': request.form['patient_first_name'],
                'patient_last_name': request.form['patient_last_name'],
                'patient_mrn': request.form.get('patient_mrn', ''),
                'patient_dob': patient_dob,
                'patient_phone': request.form.get('patient_phone', ''),
                
                # Demographics
                'age': age,
                'sex': request.form['sex'],
                
                # Clinical data
                'trestbps': int(request.form.get('systolic_bp', 120)),
                'chol': float(request.form.get('total_cholesterol', 200)),
                'hdl_cholesterol': float(request.form.get('hdl_cholesterol', 50)),
                'fbs': 1 if float(request.form.get('fasting_blood_sugar', 100)) > 120 else 0,
                'thalach': int(request.form.get('thalach', 150)),
                'smoking_status': request.form.get('smoking_status', 'never'),
                'diabetes_status': 'diabetes_status' in request.form,
                'family_history_cad': 'family_history_cad' in request.form,
                'physical_activity_level': request.form.get('physical_activity_level', 'moderate')
            }
            
            # Make prediction
            if ml_model is not None:
                # Prepare data for ML model
                ml_data = {
                    'age': data['age'],
                    'sex': 1 if data['sex'] == 'male' else 0,
                    'trestbps': data['trestbps'],
                    'chol': data['chol'],
                    'fbs': data['fbs'],
                    'thalach': data['thalach'],
                    'smoking_status': data['smoking_status'],
                    'diabetes_status': data['diabetes_status'],
                    'family_history_cad': data['family_history_cad']
                }
                prediction = ml_model.predict(ml_data)
                model_used = 'ml_model'
            else:
                # Use simple model
                simple_data = {
                    'age': data['age'],
                    'total_cholesterol': data['chol'],
                    'hdl_cholesterol': data['hdl_cholesterol'],
                    'systolic_bp': data['trestbps'],
                    'smoking_status': data['smoking_status'],
                    'diabetes_status': data['diabetes_status'],
                    'family_history_cad': data['family_history_cad']
                }
                prediction = simple_model.predict(simple_data)
                model_used = 'simple_model'
            
            # Save to database
            assessment = PatientAssessment(
                user_id=current_user.id,
                # Patient info
                patient_first_name=data['patient_first_name'],
                patient_last_name=data['patient_last_name'],
                patient_mrn=data['patient_mrn'],
                patient_dob=data['patient_dob'],
                patient_phone=data['patient_phone'],
                
                # Demographics
                age=data['age'],
                sex=data['sex'],
                
                # Clinical data
                trestbps=data['trestbps'],
                chol=data['chol'],
                hdl_cholesterol=data['hdl_cholesterol'],
                fbs=bool(data['fbs']),
                thalach=data['thalach'],
                smoking_status=data['smoking_status'],
                diabetes_status=data['diabetes_status'],
                family_history_cad=data['family_history_cad'],
                physical_activity_level=data['physical_activity_level'],
                
                # Results
                risk_category=prediction['category'],
                probability_score=prediction['probability'],
                model_used=model_used
            )
            
            db.session.add(assessment)
            db.session.commit()
            
            flash(f'Assessment completed for {assessment.full_name}', 'success')
            return redirect(url_for('results', assessment_id=assessment.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')
            return render_template('assessment.html')
    
    return render_template('assessment.html')

@app.route('/results/<int:assessment_id>')
@login_required
def results(assessment_id):
    assessment = PatientAssessment.query.get_or_404(assessment_id)
    
    if assessment.user_id != current_user.id and current_user.role != 'admin':
        return "Access denied", 403
    
    prediction = {
        'probability': assessment.probability_score,
        'category': assessment.risk_category
    }
    
    return render_template('results.html', 
                         assessment=assessment,
                         prediction=prediction,
                         now=datetime.now)

@app.route('/analytics')
@login_required
def analytics():
    """Analytics dashboard"""
    # Get all user's assessments
    assessments = PatientAssessment.query.filter_by(user_id=current_user.id)\
        .order_by(PatientAssessment.created_at).all()
    
    # Calculate statistics
    total = len(assessments)
    low_risk = sum(1 for a in assessments if a.risk_category == 'Low')
    medium_risk = sum(1 for a in assessments if a.risk_category == 'Medium')
    high_risk = sum(1 for a in assessments if a.risk_category == 'High')
    
    # Calculate risk factor correlations
    smoking_stats = calculate_risk_factor_correlation(assessments, 'smoking_status', 'current')
    diabetes_stats = calculate_risk_factor_correlation(assessments, 'diabetes_status', True)
    family_history_stats = calculate_risk_factor_correlation(assessments, 'family_history_cad', True)
    sedentary_stats = calculate_risk_factor_correlation(assessments, 'physical_activity_level', 'sedentary')
    
    # Prepare trend data (last 30 days)
    trend_data = prepare_trend_data(assessments)
    
    # Get recent assessments for activity feed
    recent_assessments = PatientAssessment.query.filter_by(user_id=current_user.id)\
        .order_by(PatientAssessment.created_at.desc())\
        .limit(10)\
        .all()
    
    stats = {
        'total_assessments': total,
        'low_risk': low_risk,
        'medium_risk': medium_risk,
        'high_risk': high_risk,
        'smoking_low': smoking_stats['low'],
        'smoking_medium': smoking_stats['medium'],
        'smoking_high': smoking_stats['high'],
        'diabetes_low': diabetes_stats['low'],
        'diabetes_medium': diabetes_stats['medium'],
        'diabetes_high': diabetes_stats['high'],
        'family_history_low': family_history_stats['low'],
        'family_history_medium': family_history_stats['medium'],
        'family_history_high': family_history_stats['high'],
        'sedentary_low': sedentary_stats['low'],
        'sedentary_medium': sedentary_stats['medium'],
        'sedentary_high': sedentary_stats['high']
    }
    
    return render_template('analytics.html',
                         user=current_user,
                         stats=stats,
                         trend_dates=trend_data['dates'],
                         trend_low=trend_data['low'],
                         trend_medium=trend_data['medium'],
                         trend_high=trend_data['high'],
                         recent_assessments=recent_assessments)

@app.route('/education')
@login_required
def education():
    """Educational resources page"""
    return render_template('education.html', user=current_user)

@app.route('/export-data')
@login_required
def export_data():
    """Export assessments as CSV"""
    assessments = PatientAssessment.query.filter_by(user_id=current_user.id)\
        .order_by(PatientAssessment.created_at.desc()).all()
    
    # Create CSV in memory
    si = StringIO()
    cw = csv.writer(si)
    
    # Write headers
    cw.writerow(['Date', 'First Name', 'Last Name', 'MRN', 'DOB', 'Phone',
                 'Age', 'Sex', 'Resting BP', 'Total Cholesterol', 'HDL Cholesterol',
                 'Fasting Blood Sugar >120', 'Max Heart Rate', 'Smoking Status', 
                 'Diabetes', 'Family History', 'Physical Activity',
                 'Risk Category', 'Probability Score', 'Model Used'])
    
    # Write data
    for a in assessments:
        cw.writerow([
            a.created_at.strftime('%Y-%m-%d %H:%M'),
            a.patient_first_name,
            a.patient_last_name,
            a.patient_mrn,
            a.patient_dob.strftime('%Y-%m-%d') if a.patient_dob else '',
            a.patient_phone,
            a.age,
            a.sex,
            a.trestbps,
            a.chol,
            a.hdl_cholesterol,
            a.fbs,
            a.thalach,
            a.smoking_status,
            a.diabetes_status,
            a.family_history_cad,
            a.physical_activity_level,
            a.risk_category,
            a.probability_score,
            a.model_used
        ])
    
    output = si.getvalue()
    si.close()
    
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=cad_assessments.csv'}
    )

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        if ml_model is not None:
            prediction = ml_model.predict(data)
            model_used = 'ml_model'
        else:
            simple_data = {
                'age': data['age'],
                'total_cholesterol': data.get('chol', 200),
                'hdl_cholesterol': 50.0,
                'systolic_bp': data.get('trestbps', 120),
                'smoking_status': data.get('smoking_status', 'never'),
                'diabetes_status': data.get('diabetes_status', False),
                'family_history_cad': data.get('family_history_cad', False)
            }
            prediction = simple_model.predict(simple_data)
            model_used = 'simple_model'
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'model_used': model_used,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
@login_required
def model_info():
    """Get information about the current model"""
    if ml_model is not None:
        return jsonify({
            'success': True,
            'model_type': ml_model.model_type,
            'performance': ml_model.performance_metrics,
            'feature_names': ml_model.feature_names
        })
    else:
        return jsonify({
            'success': True,
            'model_type': 'simple_rule_based',
            'message': 'Using simple rule-based model'
        })

@app.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for statistics"""
    try:
        assessments = PatientAssessment.query.filter_by(user_id=current_user.id).all()
        
        total = len(assessments)
        low_risk = sum(1 for a in assessments if a.risk_category == 'Low')
        medium_risk = sum(1 for a in assessments if a.risk_category == 'Medium')
        high_risk = sum(1 for a in assessments if a.risk_category == 'High')
        
        # Age distribution
        age_groups = {
            '18-30': 0, '31-40': 0, '41-50': 0, '51-60': 0, '61+': 0
        }
        
        for a in assessments:
            if a.age <= 30:
                age_groups['18-30'] += 1
            elif a.age <= 40:
                age_groups['31-40'] += 1
            elif a.age <= 50:
                age_groups['41-50'] += 1
            elif a.age <= 60:
                age_groups['51-60'] += 1
            else:
                age_groups['61+'] += 1
        
        return jsonify({
            'success': True,
            'stats': {
                'total': total,
                'low_risk': low_risk,
                'medium_risk': medium_risk,
                'high_risk': high_risk,
                'age_distribution': age_groups
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper Functions
def calculate_risk_factor_correlation(assessments, factor_name, factor_value):
    """Calculate risk distribution for a specific factor"""
    factor_assessments = [a for a in assessments if getattr(a, factor_name) == factor_value]
    total = len(factor_assessments) if factor_assessments else 1
    
    low = sum(1 for a in factor_assessments if a.risk_category == 'Low')
    medium = sum(1 for a in factor_assessments if a.risk_category == 'Medium')
    high = sum(1 for a in factor_assessments if a.risk_category == 'High')
    
    return {
        'low': round((low / total) * 100, 1),
        'medium': round((medium / total) * 100, 1),
        'high': round((high / total) * 100, 1)
    }

def prepare_trend_data(assessments):
    """Prepare data for trend charts"""
    if not assessments:
        return {'dates': [], 'low': [], 'medium': [], 'high': []}
    
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)
    
    dates = []
    low_counts = []
    medium_counts = []
    high_counts = []
    
    current = start_date
    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        
        day_assessments = [a for a in assessments 
                          if a.created_at.date() == current]
        
        low_counts.append(sum(1 for a in day_assessments if a.risk_category == 'Low'))
        medium_counts.append(sum(1 for a in day_assessments if a.risk_category == 'Medium'))
        high_counts.append(sum(1 for a in day_assessments if a.risk_category == 'High'))
        
        current += timedelta(days=1)
    
    return {
        'dates': dates,
        'low': low_counts,
        'medium': medium_counts,
        'high': high_counts
    }

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create a test user if none exists
    if not User.query.filter_by(username='doctor').first():
        test_user = User(
            username='doctor',
            email='doctor@hospital.com',
            password='password123',
            full_name='Dr. John Smith',
            role='clinician'
        )
        db.session.add(test_user)
        db.session.commit()
        print("✓ Test user created")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CAD RISK PREDICTION SYSTEM")
    print("="*60)
    print(f"✓ Flask application initialized")
    print(f"✓ Database: cad_predictions.db")
    print(f"✓ ML Model: {'Loaded' if ml_model is not None else 'Not available (using simple model)'}")
    print(f"✓ Simple Model: Available")
    print(f"\n📊 Routes available:")
    print(f"   - Home: http://localhost:5000/")
    print(f"   - Dashboard: http://localhost:5000/dashboard")
    print(f"   - Assessment: http://localhost:5000/assessment")
    print(f"   - Analytics: http://localhost:5000/analytics")
    print(f"   - Education: http://localhost:5000/education")
    print(f"\n🔐 Login credentials:")
    print(f"   Username: doctor")
    print(f"   Password: password123")
    print("="*60)
    
    app.run(debug=True, port=5000)
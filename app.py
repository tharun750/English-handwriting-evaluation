import os
import uuid
import math
import json
import pandas as pd
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from database.db import fetch_one, fetch_all, execute_query
from services.quality import check_image_quality
from services.preprocessing import preprocess_handwriting_image
from services.ocr import extract_text_from_image
from services.ai_ocr import extract_handwritten_text
from services.ai_evaluation import evaluate_handwriting
from services.feedback import generate_personalized_feedback

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'role' not in session or session['role'] != required_role:
                flash('Unauthorized access to this page.', 'danger')
                if session.get('role') == 'teacher':
                    return redirect(url_for('teacher_dashboard'))
                elif session.get('role') == 'student':
                    return redirect(url_for('student_dashboard'))
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    safe_name = os.path.basename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name)

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('role') == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash('Please enter both email and password.', 'danger')
            return render_template('login.html')

        user = fetch_one("SELECT * FROM USER WHERE email = %s", (email,))
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            session['name'] = user['name']
            session['email'] = user['email']
            session['role'] = user['role']

            if user['role'] == 'student':
                student = fetch_one("SELECT student_id FROM STUDENT WHERE user_id = %s", (user['user_id'],))
                if student:
                    session['student_id'] = student['student_id']
                flash('Welcome back! Successfully signed in as Student.', 'success')
                return redirect(url_for('student_dashboard'))
            elif user['role'] == 'teacher':
                flash('Welcome back! Successfully signed in as Teacher.', 'success')
                return redirect(url_for('teacher_dashboard'))
        else:
            flash('Invalid email address or password.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', 'student').strip()

        if not name or not email or not password:
            flash('Please fill in all required fields.', 'danger')
            return render_template('register.html')

        existing = fetch_one("SELECT user_id FROM USER WHERE email = %s", (email,))
        if existing:
            flash('Email address is already registered. Please sign in.', 'warning')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)

        if role == 'student':
            roll_number = request.form.get('roll_number', '').strip()
            department = request.form.get('department', '').strip()
            year = request.form.get('year', 1)

            if not roll_number or not department:
                flash('Please provide student roll number and department.', 'danger')
                return render_template('register.html')

            user_id = execute_query(
                "INSERT INTO USER (name, email, password, role) VALUES (%s, %s, %s, %s)",
                (name, email, hashed_password, 'student'),
                commit=True
            )
            execute_query(
                "INSERT INTO STUDENT (user_id, roll_number, department, year) VALUES (%s, %s, %s, %s)",
                (user_id, roll_number, department, year),
                commit=True
            )
        else:
            execute_query(
                "INSERT INTO USER (name, email, password, role) VALUES (%s, %s, %s, %s)",
                (name, email, hashed_password, 'teacher'),
                commit=True
            )

        flash('Registration successful! Please sign in with your credentials.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/student/dashboard')
@login_required
@role_required('student')
def student_dashboard():
    student_id = session.get('student_id')
    student = fetch_one(
        "SELECT u.name, s.roll_number, s.department, s.year FROM STUDENT s JOIN USER u ON s.user_id = u.user_id WHERE s.student_id = %s",
        (student_id,)
    )

    evaluations = fetch_all("""
        SELECT i.image_id, i.upload_date, e.score, e.grade, e.evaluation_reliability, o.ocr_status
        FROM HANDWRITTEN_IMAGE i
        JOIN HANDWRITING_FEATURES f ON i.image_id = f.image_id
        JOIN EVALUATION e ON f.feature_id = e.feature_id
        LEFT JOIN OCR_TEXT o ON i.image_id = o.image_id
        WHERE i.student_id = %s
        ORDER BY i.upload_date DESC
    """, (student_id,))

    total_evaluations = len(evaluations)
    latest_score = evaluations[0]['score'] if evaluations else None
    latest_grade = evaluations[0]['grade'] if evaluations else None
    
    scores = [e['score'] for e in evaluations]
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    return render_template(
        'student_dashboard.html',
        student=student,
        evaluations=evaluations,
        total_evaluations=total_evaluations,
        latest_score=latest_score,
        latest_grade=latest_grade,
        avg_score=avg_score
    )

@app.route('/student/upload', methods=['GET', 'POST'])
@login_required
@role_required('student')
def upload():
    if request.method == 'POST':
        if 'handwriting_image' not in request.files:
            flash('No image file selected.', 'danger')
            return redirect(request.url)

        file = request.files['handwriting_image']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            student_id = session.get('student_id')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = file.filename.rsplit('.', 1)[1].lower()
            
            unique_filename = f"student_{student_id}_{timestamp}_{uuid.uuid4().hex[:6]}.{ext}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(save_path)

            try:
              
                quality_info = check_image_quality(save_path)
                
                
                processed_filename, processed_abs_path, debug_lines_filename, debug_lines_abs_path, is_safe, preview_label = preprocess_handwriting_image(
                    save_path, app.config['UPLOAD_FOLDER']
                )
                
               
                ocr_res = extract_handwritten_text(save_path)
                extracted_text = ocr_res['text']
                ocr_conf = ocr_res['confidence']
                ocr_status = ocr_res['status']
                ocr_engine = ocr_res['engine']
                
                eval_res = evaluate_handwriting(save_path, quality_info=quality_info, ocr_conf=ocr_conf, ocr_status=ocr_status)

                score = eval_res['overall_score']
                grade = eval_res['grade']
                reliability = eval_res.get('evaluation_reliability', 'High')
                
                feedback_text = eval_res.get('summary', 'Evaluation completed.')
                strengths = eval_res.get('strengths', [])
                improvements = eval_res.get('improvements', [])
                
                strengths_str = "\n".join(strengths) if strengths else ""
                improvements_str = "\n".join(improvements) if improvements else ""

                if improvements:
                    suggestion = "Targeted Practice Suggestions: " + " ".join([f"({idx+1}) {imp}" for idx, imp in enumerate(improvements)])
                else:
                    suggestion = "Maintain your regular handwriting posture, pen grip, and steady baseline practice!"

                image_id = execute_query(
                    """INSERT INTO HANDWRITTEN_IMAGE 
                       (student_id, image_path, processed_image_path, debug_lines_path, image_quality_score, image_quality_status) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (student_id, unique_filename, processed_filename, debug_lines_filename, quality_info['quality_score'], quality_info['quality_status']),
                    commit=True
                )
                
                execute_query(
                    "INSERT INTO OCR_TEXT (image_id, extracted_text, ocr_confidence, ocr_status, engine) VALUES (%s, %s, %s, %s, %s)",
                    (image_id, extracted_text, ocr_conf, ocr_status, ocr_engine),
                    commit=True
                )
                
                feature_id = execute_query(
                    """INSERT INTO HANDWRITING_FEATURES 
                       (image_id, letter_formation, word_spacing, alignment, slant, size_consistency, legibility) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (image_id, eval_res['letter_formation'], eval_res['word_spacing'], 
                     eval_res['alignment'], eval_res['slant_consistency'], eval_res['size_consistency'], eval_res['legibility']),
                    commit=True
                )
                
                evaluation_id = execute_query(
                    "INSERT INTO EVALUATION (feature_id, score, grade, evaluation_reliability) VALUES (%s, %s, %s, %s)",
                    (feature_id, score, grade, reliability),
                    commit=True
                )
                
                execute_query(
                    "INSERT INTO FEEDBACK (evaluation_id, feedback_text, suggestion, strengths, improvements) VALUES (%s, %s, %s, %s, %s)",
                    (evaluation_id, feedback_text, suggestion, strengths_str, improvements_str),
                    commit=True
                )

                flash('Handwriting sample evaluated successfully using Gemini Vision AI!', 'success')
                return redirect(url_for('result', image_id=image_id))

            except Exception as e:
                print(f"Pipeline Error: {e}")
                flash(f"AI evaluation is temporarily unavailable. Please try again. ({str(e)})", 'danger')
                return redirect(request.url)

        else:
            flash('Invalid file extension. Please upload a .jpg, .jpeg, or .png image.', 'danger')

    return render_template('upload.html')

@app.route('/student/result/<int:image_id>')
@login_required
def result(image_id):
    sql = """
        SELECT 
            i.image_id, i.image_path, i.processed_image_path, i.debug_lines_path, i.image_quality_score, i.image_quality_status, i.upload_date,
            o.extracted_text, o.ocr_confidence, o.ocr_status, o.engine as ocr_engine,
            f.letter_formation, f.word_spacing, f.alignment, f.slant, f.size_consistency, f.legibility,
            e.score, e.grade, e.evaluation_reliability,
            fb.feedback_text, fb.suggestion, fb.strengths, fb.improvements,
            u.name as student_name, s.roll_number, s.student_id
        FROM HANDWRITTEN_IMAGE i
        JOIN STUDENT s ON i.student_id = s.student_id
        JOIN USER u ON s.user_id = u.user_id
        LEFT JOIN OCR_TEXT o ON i.image_id = o.image_id
        LEFT JOIN HANDWRITING_FEATURES f ON i.image_id = f.image_id
        LEFT JOIN EVALUATION e ON f.feature_id = e.feature_id
        LEFT JOIN FEEDBACK fb ON e.evaluation_id = fb.evaluation_id
        WHERE i.image_id = %s
    """

    res = fetch_one(sql, (image_id,))
    if not res:
        flash('Evaluation record not found.', 'danger')
        return redirect(url_for('student_dashboard'))

    if session.get('role') == 'student' and res['student_id'] != session.get('student_id'):
        flash('Access denied to requested evaluation record.', 'danger')
        return redirect(url_for('student_dashboard'))

    strengths_list = res['strengths'].split("\n") if res.get('strengths') else []
    improvements_list = res['improvements'].split("\n") if res.get('improvements') else []

    student = {'name': res['student_name'], 'roll_number': res['roll_number']}
    return render_template(
        'result.html',
        result=res,
        student=student,
        strengths_list=strengths_list,
        improvements_list=improvements_list
    )

@app.route('/student/history')
@login_required
@role_required('student')
def history():
    student_id = session.get('student_id')
    sql = """
        SELECT i.image_id, i.upload_date, i.image_quality_status,
               o.extracted_text, o.ocr_status, o.ocr_confidence,
               e.score, e.grade, e.evaluation_reliability
        FROM HANDWRITTEN_IMAGE i
        LEFT JOIN OCR_TEXT o ON i.image_id = o.image_id
        LEFT JOIN HANDWRITING_FEATURES f ON i.image_id = f.image_id
        LEFT JOIN EVALUATION e ON f.feature_id = e.feature_id
        WHERE i.student_id = %s
        ORDER BY i.upload_date DESC
    """
    evaluations = fetch_all(sql, (student_id,))
    return render_template('history.html', evaluations=evaluations)

@app.route('/student/progress')
@login_required
@role_required('student')
def progress():
    student_id = session.get('student_id')
    sql = """
        SELECT i.upload_date, e.score, e.grade, e.evaluation_reliability
        FROM HANDWRITTEN_IMAGE i
        JOIN HANDWRITING_FEATURES f ON i.image_id = f.image_id
        JOIN EVALUATION e ON f.feature_id = e.feature_id
        WHERE i.student_id = %s
        ORDER BY i.upload_date ASC
    """
    evaluations = fetch_all(sql, (student_id,))
    
    scores = [e['score'] for e in evaluations]
    first_score = scores[0] if scores else None
    latest_score = scores[-1] if scores else None
    improvement = round(latest_score - first_score, 2) if (first_score is not None and latest_score is not None) else 0.0
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    return render_template(
        'progress.html',
        evaluations=evaluations,
        first_score=first_score,
        latest_score=latest_score,
        improvement=improvement,
        avg_score=avg_score
    )


@app.route('/teacher/dashboard')
@login_required
@role_required('teacher')
def teacher_dashboard():
    students = fetch_all("SELECT student_id FROM STUDENT")
    total_students = len(students)

    recent_evaluations = fetch_all("""
        SELECT i.image_id, i.student_id, u.name as student_name, s.roll_number, s.department,
               i.upload_date, e.score, e.grade, e.evaluation_reliability, o.ocr_status
        FROM HANDWRITTEN_IMAGE i
        JOIN STUDENT s ON i.student_id = s.student_id
        JOIN USER u ON s.user_id = u.user_id
        JOIN HANDWRITING_FEATURES f ON i.image_id = f.image_id
        JOIN EVALUATION e ON f.feature_id = e.feature_id
        LEFT JOIN OCR_TEXT o ON i.image_id = o.image_id
        ORDER BY i.upload_date DESC
        LIMIT 10
    """)

    all_scores = fetch_all("SELECT score FROM EVALUATION")
    total_evaluations = len(all_scores)
    class_avg = round(sum([s['score'] for s in all_scores]) / total_evaluations, 2) if total_evaluations > 0 else None

    return render_template(
        'teacher_dashboard.html',
        total_students=total_students,
        total_evaluations=total_evaluations,
        class_avg=class_avg,
        recent_evaluations=recent_evaluations
    )

@app.route('/teacher/students')
@login_required
@role_required('teacher')
def teacher_students():
    sql = """
        SELECT s.student_id, s.roll_number, s.department, s.year, u.name, u.email,
               COUNT(i.image_id) as total_evaluations,
               AVG(e.score) as avg_score
        FROM STUDENT s
        JOIN USER u ON s.user_id = u.user_id
        LEFT JOIN HANDWRITTEN_IMAGE i ON s.student_id = i.student_id
        LEFT JOIN HANDWRITING_FEATURES f ON i.image_id = f.image_id
        LEFT JOIN EVALUATION e ON f.feature_id = e.feature_id
        GROUP BY s.student_id, s.roll_number, s.department, s.year, u.name, u.email
        ORDER BY u.name ASC
    """
    students = fetch_all(sql)
    return render_template('students.html', students=students)

@app.route('/teacher/student/<int:student_id>')
@login_required
@role_required('teacher')
def teacher_student_details(student_id):
    student = fetch_one(
        "SELECT s.student_id, s.roll_number, s.department, s.year, u.name, u.email FROM STUDENT s JOIN USER u ON s.user_id = u.user_id WHERE s.student_id = %s",
        (student_id,)
    )
    if not student:
        flash('Student record not found.', 'danger')
        return redirect(url_for('teacher_students'))

    evaluations = fetch_all("""
        SELECT i.image_id, i.upload_date, o.extracted_text, o.ocr_status, o.ocr_confidence,
               f.letter_formation, f.word_spacing, f.alignment, f.slant, f.size_consistency, f.legibility,
               e.score, e.grade, e.evaluation_reliability, fb.feedback_text
        FROM HANDWRITTEN_IMAGE i
        LEFT JOIN OCR_TEXT o ON i.image_id = o.image_id
        LEFT JOIN HANDWRITING_FEATURES f ON i.image_id = f.image_id
        LEFT JOIN EVALUATION e ON f.feature_id = e.feature_id
        LEFT JOIN FEEDBACK fb ON e.evaluation_id = fb.evaluation_id
        WHERE i.student_id = %s
        ORDER BY i.upload_date DESC
    """, (student_id,))

    return render_template('student_details.html', student=student, evaluations=evaluations)

if __name__ == '__main__':
    print("Starting English Handwriting Evaluation System...")
    app.run(host='127.0.0.1', port=5000, debug=Config.DEBUG)

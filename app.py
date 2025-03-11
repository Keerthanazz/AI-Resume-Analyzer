from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
import logging
import io
import re
from flask_session import Session  # For persistent session storage
import threading  # Added for background processing
import random  # For API key rotation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Multiple API keys for rotation to prevent quota exhaustion
GEMINI_API_KEYS = [
    "AAIzaSyBHk1ygR5cRXxJJn7zPemh9UuEvFRyjxuM",
    "AIzaSyBSvvfD8r0kdwX1v8sLG7VfsQTPOHVpmug",  # Add your additional API keys here
    "AIzaSyBWfRI3_vBTS1vO8rpE22-iOgud9aEPKA0",
    "AIzaSyDt33KzhMmCjaUnnpKPi9efmpAoekogr70",
    "AIzaSyCWMKT-K4ITpVnJZtXWuHAnNpIDMAWnjx8"
]

# Initialize Flask app
app = Flask(__name__)

# Secure session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.secret_key = "your_secret_key_here"  # Change this to a secure key

Session(app)  # Initialize Flask session

# Global dictionary to track background tasks
background_tasks = {}

def get_random_api_key():
    """Returns a randomly selected API key from the available pool"""
    return random.choice(GEMINI_API_KEYS)

def clean_ocr_text(text):
    """Cleans OCR text by removing unwanted symbols and fixing common issues"""
    # Remove non-standard characters but keep basic punctuation
    cleaned_text = re.sub(r'[^\w\s.,;:?!()@-]', ' ', text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Fix common OCR errors (can be expanded)
    cleaned_text = cleaned_text.replace('l l', 'll')
    cleaned_text = cleaned_text.replace('I l', 'Il')
    return cleaned_text.strip()

class JobApplicationAnalyzer:
    def __init__(self):
        # Initialize without API key - we'll configure it per request
        self.timeout = 30  # 30 seconds timeout for API calls

    def get_model(self):
        """Returns a configured model with a fresh API key"""
        api_key = get_random_api_key()
        genai.configure(api_key=api_key)
        logger.info(f"Using API key ending with ...{api_key[-4:]}")
        
        return genai.GenerativeModel(
            'gemini-1.5-pro',
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40
            }
        )

    def extract_text_from_pdf(self, uploaded_file):
        """ Extracts text from a PDF file using OCR and cleans the output """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name

            images = convert_from_path(temp_pdf_path)
            extracted_text = []
            
            for img in images:
                img_array = np.array(img)
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                text = pytesseract.image_to_string(gray)
                # Clean OCR text
                text = clean_ocr_text(text)
                extracted_text.append(text)

            os.remove(temp_pdf_path)
            return "\n\n".join(extracted_text).strip()
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            return f"Error extracting PDF: {str(e)}"

    def initial_resume_analysis(self, resume_text):
        """ Performs initial resume analysis with cleaned text """
        try:
            model = self.get_model()  # Get fresh model with rotated API key
            
            # Use a more concise prompt with instruction to ignore unwanted symbols
            prompt = f"""
            Analyze this resume briefly (max 300 words). Focus only on legitimate resume content 
            and ignore any OCR artifacts or strange symbols:
            do not use symbols like ** dont us e any symbols, , give the initial analysis in points points and also teh topics in bold terms
            
            Resume:
            {resume_text[:2000]}  # Limit input size
            
            Format your analysis with clear sections. DO NOT include any strange symbols or characters in your output.
            Focus only on providing a professional assessment of the candidate's qualifications.
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error in initial analysis: {str(e)}")
            return f"Error in initial analysis: {str(e)}"

    def analyze_job_compatibility(self, resume_text, job_description, company_name, role_title=""):
        """ Compares resume with job requirements and returns a report """
        try:
            model = self.get_model()  # Get fresh model with rotated API key
            
            # More concise prompt with length limit and explicit instruction about formatting
            prompt = f"""
            Analyze resume compatibility with job description (max 500 words):
            
            1. Overall match score (0-100%)
            2. Top 3 matching skills/qualifications
            3. Top 3 missing skills/qualifications
            
            Company: {company_name}
            Role: {role_title}
            Job Description:
            {job_description[:1000]}  # Limit input size
            
            Resume:
            {resume_text[:2000]}  # Limit input size
            
            IMPORTANT FORMATTING INSTRUCTIONS:
            - DO NOT include any special characters or symbols in your analysis
            - Format your response in plain text with clear headings
            - Focus on professional assessment
            - Use only standard punctuation (periods, commas, etc.)
            - Be direct and clear in your assessment
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing job compatibility: {str(e)}")
            return f"Error analyzing job compatibility: {str(e)}"

    def create_skill_enhancement_plan(self, resume_text, job_description, company_name, role_title=""):
        """ Creates a detailed skill enhancement plan with courses and resources """
        try:
            model = self.get_model()  # Get fresh model with rotated API key
            
            prompt = f"""
            Create a detailed skill enhancement plan for this job seeker. Based on their resume and the job description, identify skill gaps and create:
            please provide in tabular column rad map, with all respective courses links , teht are all available to enhance the user skills, and make sure if the use teh courses aws certifiaction and comapre with teh skills and certificate they alrady have
            1. A prioritized list of skills to develop (focus on the 3-5 most important missing skills)
            2. For each skill:
               - A learning roadmap with steps from beginner to job-ready
               - Recommended online courses (include specific course names from platforms like Coursera, Udemy, edX, LinkedIn Learning)
               - Free resources and tutorials (include specific links or names of YouTube channels, websites, etc.)
               - Books or documentation that would be helpful
               - Practical projects to build for the portfolio that demonstrate this skill
            3. Estimated timeline to acquire each skill (in weeks or months)
            4. How to demonstrate these skills on resume and in interviews
            
            Format the output with clear headers and organized sections.
            
            IMPORTANT FORMATTING INSTRUCTIONS:
            - DO NOT include any strange symbols or characters in your output
            - Use only standard punctuation (periods, commas, etc.)
            - Format your response in plain text with clear headings
            - Use numbers and bullet points for clear organization
            
            Company: {company_name}
            Role: {role_title}
            Job Description:
            {job_description[:1000]}
            
            Resume:
            {resume_text[:2000]}
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error creating skill enhancement plan: {str(e)}")
            return f"Error creating skill enhancement plan: {str(e)}"
    def create_optimized_resume(self, resume_text, job_description, company_name, role_title=""):
        """ Creates an optimized version of the resume for the specific job using only data from the original resume """
        try:
            model = self.get_model()  # Get fresh model with rotated API key
                    
            # Improved prompt that emphasizes preserving original content and structure
            prompt = f"""
            Create an optimized version of the resume below, tailored for the job at {company_name} for the {role_title} position.
                    
            INSTRUCTIONS:
            ATS Based correct resume for that role 
            1. ONLY use information that exists in the original resume - do not invent or add new qualifications, experiences, or skills
            2. Preserve the original resume structure (sections, format) and personal information exactly as provided
            3. Reword existing content to emphasize relevant skills and experiences for the job description
            4. Improve clarity where OCR may have introduced errors, but preserve all factual content
            5. Do not add generic text like "[Your Address]" or placeholders - only use data directly from the resume
            6. Format the optimized resume in a clean, professional way while maintaining the original information
                    
            Company: {company_name}
            Role: {role_title}
            Job Description:
            {job_description[:1000]}
                    
            Original Resume:
            {resume_text[:4000]}
                    
            Remember: ONLY use information that appears in the original resume above. Do not fabricate or add generic information.
            """
                    
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error creating optimized resume: {str(e)}")
            return f"Error creating optimized resume: {str(e)}"
def create_optimized_resume(self, resume_text, job_description, company_name, role_title=""):
    """ Creates an optimized version of the resume for the specific job using only data from the original resume """
    try:
        model = self.get_model()  # Get fresh model with rotated API key
        
        # Improved prompt that emphasizes preserving original content and structure
        prompt = f"""
        Create an optimized version of the resume below, tailored for the job at {company_name} for the {role_title} position.
        
        INSTRUCTIONS:
        ATS Based correct resume for that role 
        1. ONLY use information that exists in the original resume - do not invent or add new qualifications, experiences, or skills
        2. Preserve the original resume structure (sections, format) and personal information exactly as provided
        3. Reword existing content to emphasize relevant skills and experiences for the job description
        4. Improve clarity where OCR may have introduced errors, but preserve all factual content
        5. Do not add generic text like "[Your Address]" or placeholders - only use data directly from the resume
        6. Format the optimized resume in a clean, professional way while maintaining the original information
        
        Company: {company_name}
        Role: {role_title}
        Job Description:
        {job_description[:1000]}
        
        Original Resume:
        {resume_text[:4000]}
        
        Remember: ONLY use information that appears in the original resume above. Do not fabricate or add generic information.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error creating optimized resume: {str(e)}")
        return f"Error creating optimized resume: {str(e)}"

# Background task function for job compatibility analysis
def process_job_analysis(user_id, resume_text, job_description, company_name, role_title):
    try:
        background_tasks[user_id]['status'] = 'processing'
        
        # Do the job analysis
        job_analysis = analyzer.analyze_job_compatibility(
            resume_text, job_description, company_name, role_title
        )
        
        # Store in the background tasks dict
        background_tasks[user_id]['job_analysis'] = job_analysis
        background_tasks[user_id]['status'] = 'completed'
        logger.info(f"Completed job analysis for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error in background job analysis: {str(e)}")
        background_tasks[user_id]['status'] = 'error'
        background_tasks[user_id]['error'] = str(e)

# Background task function for skill enhancement plan
def process_skill_plan(user_id, resume_text, job_description, company_name, role_title):
    try:
        background_tasks[user_id]['skill_plan_status'] = 'processing'
        
        # Generate the skill enhancement plan
        skill_plan = analyzer.create_skill_enhancement_plan(
            resume_text, job_description, company_name, role_title
        )
        
        # Store in the background tasks dict
        background_tasks[user_id]['skill_enhancement_plan'] = skill_plan
        background_tasks[user_id]['skill_plan_status'] = 'completed'
        logger.info(f"Completed skill enhancement plan for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error in background skill plan creation: {str(e)}")
        background_tasks[user_id]['skill_plan_status'] = 'error'
        background_tasks[user_id]['skill_plan_error'] = str(e)

# Initialize the analyzer
analyzer = JobApplicationAnalyzer()

@app.route('/')
def index():
    return redirect(url_for('resume_analysis'))

@app.route('/resume_analysis', methods=['GET', 'POST'])
def resume_analysis():
    """ Handles resume uploads and analysis """
    logger.info(f"Session keys at resume_analysis: {list(session.keys())}")

    if request.method == 'POST':
        if 'resume_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['resume_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and file.filename.endswith('.pdf'):
            resume_text = analyzer.extract_text_from_pdf(file)
            
            if not resume_text.startswith("Error"):
                session['resume_text'] = resume_text
                session['initial_analysis'] = analyzer.initial_resume_analysis(resume_text)
                session.modified = True  # Ensure session updates
                logger.info("Resume text saved to session successfully")
                flash('Resume analyzed successfully!')
            else:
                flash(resume_text)

    return render_template('resume_analysis.html', 
                           resume_text=session.get('resume_text', ''), 
                           initial_analysis=session.get('initial_analysis', ''))

@app.route('/job_compatibility', methods=['GET', 'POST'])
def job_compatibility():
    """ Compares the analyzed resume with a job description """
    logger.info(f"Job compatibility route accessed. Session keys: {list(session.keys())}")
    user_id = session.get('_id', 'default_user')

    # Check if resume is analyzed
    resume_text = session.get('resume_text')
    if not resume_text:
        flash('Please upload and analyze your resume first')
        return redirect(url_for('resume_analysis'))

    # Check for ongoing background task
    task_status = None
    if user_id in background_tasks:
        task_status = background_tasks[user_id].get('status')
        
        # If completed, update session with results
        if task_status == 'completed':
            session['job_analysis'] = background_tasks[user_id]['job_analysis']
            session.modified = True
            
        # If error, show error message
        elif task_status == 'error':
            flash(f"Error in analysis: {background_tasks[user_id].get('error', 'Unknown error')}")

    if request.method == 'POST':
        company_name = request.form.get('company_name', '')
        role_title = request.form.get('role_title', '')
        job_description = request.form.get('job_description', '')

        if company_name and job_description:
            session['company_name'] = company_name
            session['role_title'] = role_title
            session['job_description'] = job_description
            session.modified = True
            
            # Set up background task for job analysis
            if user_id not in background_tasks:
                background_tasks[user_id] = {}
            
            background_tasks[user_id]['status'] = 'pending'
            
            # Start background thread for analysis
            thread = threading.Thread(
                target=process_job_analysis,
                args=(user_id, resume_text, job_description, company_name, role_title)
            )
            thread.daemon = True
            thread.start()
            
            flash('Job compatibility analysis started! Results will appear shortly...')
            
            # Immediately return the template with processing status
            return render_template('job_compatibility.html',
                                company_name=company_name,
                                role_title=role_title,
                                job_description=job_description,
                                processing=True,
                                job_analysis=None,
                                has_optimized_resume=False,
                                has_skill_plan='skill_enhancement_plan' in session)

    # Return template with current status
    return render_template('job_compatibility.html',
                        company_name=session.get('company_name', ''),
                        role_title=session.get('role_title', ''),
                        job_description=session.get('job_description', ''),
                        processing=(task_status == 'processing'),
                        job_analysis=session.get('job_analysis', ''),
                        has_optimized_resume='optimized_resume' in session,
                        has_skill_plan='skill_enhancement_plan' in session)

@app.route('/get_analysis_status')
def get_analysis_status():
    """AJAX endpoint to check analysis status"""
    user_id = session.get('_id', 'default_user')
    
    if user_id in background_tasks:
        status = background_tasks[user_id].get('status')
        
        if status == 'completed':
            return {'status': status, 'result': background_tasks[user_id]['job_analysis']}
        elif status == 'error':
            return {'status': 'error', 'message': background_tasks[user_id].get('error', 'Unknown error')}
        else:
            return {'status': status}
    
    return {'status': 'not_started'}

@app.route('/get_skill_plan_status')
def get_skill_plan_status():
    """AJAX endpoint to check skill plan status"""
    user_id = session.get('_id', 'default_user')
    
    if user_id in background_tasks:
        status = background_tasks[user_id].get('skill_plan_status')
        
        if status == 'completed':
            return {'status': status, 'result': background_tasks[user_id]['skill_enhancement_plan']}
        elif status == 'error':
            return {'status': 'error', 'message': background_tasks[user_id].get('skill_plan_error', 'Unknown error')}
        else:
            return {'status': status}
    
    return {'status': 'not_started'}

@app.route('/skill_enhancement', methods=['GET', 'POST'])
def skill_enhancement():
    """Handles the skill enhancement plan generation and display"""
    user_id = session.get('_id', 'default_user')
    
    # Check if job compatibility analysis is done
    if 'job_analysis' not in session:
        flash('Please complete job compatibility analysis first')
        return redirect(url_for('job_compatibility'))
    
    resume_text = session.get('resume_text')
    company_name = session.get('company_name')
    role_title = session.get('role_title')
    job_description = session.get('job_description')
    
    # Check for ongoing skill plan generation
    skill_plan_status = None
    if user_id in background_tasks:
        skill_plan_status = background_tasks[user_id].get('skill_plan_status')
        
        # If completed, update session with results
        if skill_plan_status == 'completed':
            session['skill_enhancement_plan'] = background_tasks[user_id]['skill_enhancement_plan']
            session.modified = True
        
        # If error, show error message
        elif skill_plan_status == 'error':
            flash(f"Error creating skill plan: {background_tasks[user_id].get('skill_plan_error', 'Unknown error')}")
    
    # Handle POST request to generate skill plan
    if request.method == 'POST':
        if resume_text and company_name and job_description:
            # Initialize the background tasks dict for this user if needed
            if user_id not in background_tasks:
                background_tasks[user_id] = {}
            
            background_tasks[user_id]['skill_plan_status'] = 'pending'
            
            # Start background thread for skill plan generation
            thread = threading.Thread(
                target=process_skill_plan,
                args=(user_id, resume_text, job_description, company_name, role_title)
            )
            thread.daemon = True
            thread.start()
            
            flash('Skill enhancement plan generation started! Results will appear shortly...')
            
            # Return template with processing status
            return render_template('skill_enhancement.html',
                                company_name=company_name,
                                role_title=role_title,
                                processing=True,
                                skill_plan=None)
    
    # Return template with current skill plan or processing status
    return render_template('skill_enhancement.html',
                        company_name=session.get('company_name', ''),
                        role_title=session.get('role_title', ''),
                        processing=(skill_plan_status == 'processing'),
                        skill_plan=session.get('skill_enhancement_plan', ''))

@app.route('/download_skill_plan')
def download_skill_plan():
    """Downloads the skill enhancement plan as a text file"""
    skill_plan = session.get('skill_enhancement_plan')
    company_name = session.get('company_name', 'company')
    role_title = session.get('role_title', 'role')
    
    if not skill_plan:
        flash('No skill enhancement plan available')
        return redirect(url_for('skill_enhancement'))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"skill_plan_{company_name.replace(' ', '_')}_{role_title.replace(' ', '_')}_{timestamp}.txt"
    
    return send_file(io.BytesIO(skill_plan.encode()), mimetype='text/plain', as_attachment=True, download_name=filename)

@app.route('/optimize_resume', methods=['POST'])
def optimize_resume():
    """Separate endpoint to generate optimized resume"""
    resume_text = session.get('resume_text')
    company_name = session.get('company_name', '')
    role_title = session.get('role_title', '')
    job_description = session.get('job_description', '')
    
    if not all([resume_text, company_name, job_description]):
        flash('Missing required information')
        return redirect(url_for('job_compatibility'))
    
    try:
        optimized = analyzer.create_optimized_resume(
            resume_text, job_description, company_name, role_title
        )
        session['optimized_resume'] = optimized
        session.modified = True
        flash('Resume optimized successfully!')
    except Exception as e:
        flash(f'Error optimizing resume: {str(e)}')
    
    return redirect(url_for('job_compatibility'))

@app.route('/download_resume')
def download_resume():
    """ Downloads the optimized resume as a text file """
    optimized_resume = session.get('optimized_resume')
    company_name = session.get('company_name', 'company')
    role_title = session.get('role_title', 'role')

    if not optimized_resume:
        flash('No optimized resume available')
        return redirect(url_for('job_compatibility'))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimized_resume_{company_name.replace(' ', '_')}_{role_title.replace(' ', '_')}_{timestamp}.txt"

    return send_file(io.BytesIO(optimized_resume.encode()), mimetype='text/plain', as_attachment=True, download_name=filename)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/clear_session')
def clear_session():
    """ Clears the session data for debugging """
    user_id = session.get('_id', 'default_user')
    if user_id in background_tasks:
        del background_tasks[user_id]
    session.clear()
    flash('Session cleared')
    return redirect(url_for('resume_analysis'))

@app.route('/debug_session')
def debug_session():
    """ Debugging route to check session contents """
    user_id = session.get('_id', 'default_user')
    task_info = background_tasks.get(user_id, {})
    return f"Session contents: {dict(session)}<br>Task info: {task_info}"

if __name__ == '__main__':
    app.run(debug=True)

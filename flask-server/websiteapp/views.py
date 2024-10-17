from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, session, send_file, jsonify
from flask_login import login_required, current_user, logout_user
from .models import FileUpload
from . import db
import os
import msal
import time
import base64
import io
import re
import matplotlib.pyplot as plt
import matplotlib
from werkzeug.utils import secure_filename
import gradio as gr

matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments

views = Blueprint('views', __name__)

# Define MSAL configurations
CLIENT_ID = '3fc2c938-3b35-4a5f-bb14-a31096aa4a24'  # Application (client) ID
CLIENT_SECRET = 'iIK8Q~vYEyE1m0dY39DJJCRkbyXOYuoBSauT3bg6'  # client secret 
TENANT_ID = 'bbc0b28d-7832-4d95-b452-52bbf58bbcc9'  # Directory (tenant) ID
AUTHORITY = 'https://login.microsoftonline.com/common'   # Authority URL
REDIRECT_URI = 'http://localhost:5000/redirect'  # Ensure this matches its registered redirect URI
SCOPE = ["User.Read"]  # Scope for MS Graph API

# Create an MSAL client instance
msal_client = msal.ConfidentialClientApplication(
    CLIENT_ID, 
    authority=AUTHORITY, 
    client_credential=CLIENT_SECRET
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'txt'}

def allowed_file(filename):
    """Checking if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(filepath):
    """Process the uploaded file and extract its data."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)

def is_token_expired():
    """Check if the access token is expired."""
    if 'expires_at' in session:
        return time.time() > session['expires_at']
    return True

@views.route('/')
def home():
    if 'access_token' in session and not is_token_expired():
        return render_template("home.html", user=current_user)
    else:
        flash('Your session has expired. Please log in again.', category='warning')
        return redirect(url_for('views.login'))

@views.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'access_token' not in session or is_token_expired():
        flash('Your session has expired. Please log in again.', category='warning')
        return redirect(url_for('views.login'))  # Redirect to login if not authenticated

    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            # Ensure the uploads directory exists
            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)  # Create the directory if it doesn't exist
            
            # Save file to 'uploads/' folder
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            # Save only the file path in session
            session['file_path'] = filepath
            session['filename'] = filename  
            
            # Process the file into a DataFrame
            file_data = process_file(filepath)

            # Process the file and extract metadata for display
            metadata = {
                'columns': list(file_data.columns),
                'data_types': file_data.dtypes.astype(str).to_dict(),
                'num_rows': len(file_data)
            }
            session['metadata'] = metadata
            flash('File uploaded and processed successfully!', category='success')
            return redirect(url_for('views.visualize'))

        flash('Invalid file. Please upload a valid .csv or .xlsx file.', category='error')

    return render_template('upload.html')

def process_query(user_query, file_data):
    """Process the user query and return the response and plot."""
    # Initialize the LangChain agent
    llm = OpenAI(temperature=0, openai_api_key='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')  # Replace with your actual API key
    agent = create_pandas_dataframe_agent(llm, file_data, verbose=True, allow_dangerous_code=True)

    # Run the user query through the agent
    response = str(agent.invoke(user_query))

    # Check for specific queries and generate plots
    if "distribution of malignant and benign cases" in user_query.lower():
        counts = file_data['diagnosis'].value_counts()
        plt.bar(counts.index, counts.values)
        plt.xlabel('Diagnosis')
        plt.ylabel('Number of Cases')
        plt.title('Distribution of Malignant and Benign Cases')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()

        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return response, img_base64  # Return both the text response and the image data

    return response, None  # If no image, return just the response

def launch_gradio_interface(file_path):
    """Launch the Gradio interface for processing queries."""
    # Load the data for Gradio interface
    file_data = process_file(file_path)

    # Create Gradio interface
    iface = gr.Interface(
        fn=lambda query: process_query(query, file_data),  # Function to call
        inputs=gr.Textbox(label="Enter your query"),  # Input textbox
        outputs=[gr.Textbox(label="Response"), gr.Image(type="numpy", label="Plot")],  # Text and image outputs
        title="Data Visualization Chatbot",
        description="Ask questions about the dataset."
    )
    iface.launch()  # Launch the interface

@views.route('/gradio', methods=['GET'])
def gradio():
    """Route to launch the Gradio interface."""
    if 'file_path' in session:
        launch_gradio_interface(session['file_path'])
        return redirect(url_for('views.upload'))  # Redirect to upload page after Gradio runs
    else:
        flash('No data to visualize. Please upload a file.', category='warning')
        return redirect(url_for('views.upload'))

@views.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if 'file_path' not in session:
        flash('No data to visualize. Please upload a file.', category='warning')
        return redirect(url_for('views.upload'))

    file_path = session['file_path']
    file_data = process_file(file_path)

    responses = []
    img_base64s = []

    if request.method == 'POST':
        user_query = request.form.get('query')

        if user_query:
            llm = OpenAI(temperature=0, openai_api_key='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')  # Replace with your actual API key
            agent = create_pandas_dataframe_agent(llm, file_data, verbose=True, allow_dangerous_code=True)

            try:
                # Run user query through LangChain agent
                response = str(agent.invoke(user_query))  # This calls the agent to process the query
                responses.append(response)

                # Extract and execute code blocks from response
                code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)

                # Handle code execution and plotting
                img_base64 = execute_code_blocks(code_blocks)
                img_base64s.append(img_base64)

                if img_base64:
                    session['img_buffer'] = img_base64  # Store image buffer for downloading
                else:
                    flash('No image generated from the query.', category='warning')

            except Exception as e:
                response = f"Error: {str(e)}"
                flash(response, category='error')

    combined_responses = list(zip(responses, img_base64s))
    
    return render_template('visualize.html', combined_responses=combined_responses)

def execute_code_blocks(code_blocks):
    img_base64 = None
    for code_block in code_blocks:
        try:
            exec(code_block)  # Execute the code block safely
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()  # Close the plot to free memory
        except Exception as e:
            print(f"Error executing code block: {e}")  # Log any error
            return None
    return img_base64

@views.route('/download_image', methods=['POST'])
def download_image():
    format_type = request.form.get('format')

    if 'img_buffer' in session:
        img_buffer = io.BytesIO(base64.b64decode(session['img_buffer']))

        if format_type == 'png':
            return send_file(img_buffer, as_attachment=True, download_name='plot.png', mimetype='image/png')
        elif format_type == 'pdf':
            return send_file(img_buffer, as_attachment=True, download_name='plot.pdf', mimetype='application/pdf')
        elif format_type == 'svg':
            return send_file(img_buffer, as_attachment=True, download_name='plot.svg', mimetype='image/svg+xml')
    else:
        return "No image found in session", 400

@views.route("/members")
def members():
    return {"members": ["Member1", "Member2", "Member3"]}

@views.route('/login')
def login():
    # Build the login URL
    auth_url = msal_client.get_authorization_request_url(
        scopes=SCOPE,
        redirect_uri=REDIRECT_URI
    )
    return redirect(auth_url)

@views.route('/redirect')
def get_a_token():
    code = request.args.get('code')  # Get authorization code from the query string
    if not code:
        flash('No code provided', category='error')
        return redirect(url_for('views.home'))

    # Exchange the authorization code for a token
    token_response = msal_client.acquire_token_by_authorization_code(
        code,
        scopes=SCOPE,
        redirect_uri=REDIRECT_URI
    )

    if 'access_token' in token_response:
        session['access_token'] = token_response['access_token']  # Store the token in the session
        session['expires_at'] = time.time() + token_response['expires_in']  # Store expiration time
        flash('Login successful!', category='success')
        return redirect(url_for('views.home'))
    else:
        error_message = token_response.get('error_description', 'Unknown error occurred')
        flash(f'Login failed: {error_message}', category='error')
        return redirect(url_for('views.home'))

@views.route('/logout')
def logout():
    logout_user()  # Log out the user
    session.clear()  # Clear session
    flash('You have been logged out.', category='info')
    return redirect(url_for('views.home'))
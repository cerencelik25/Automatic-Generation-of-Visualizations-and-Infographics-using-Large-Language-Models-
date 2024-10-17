from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from os import path, getenv
from flask_login import LoginManager, current_user
from msal import ConfidentialClientApplication
from flask_cors import CORS 

db = SQLAlchemy()
migrate = Migrate()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'asssasa'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['UPLOAD_FOLDER'] = 'uploads' 

    # MSAL Configuration
    app.config['CLIENT_ID'] = 'your-client-id'
    app.config['CLIENT_SECRET'] = 'your-client-secret'
    app.config['AUTHORITY'] = 'https://login.microsoftonline.com/your-tenant-id'
    app.config['REDIRECT_URI'] = 'http://localhost:5000/getAToken' 

    # OpenAI API Key
    app.config['OPENAI_API_KEY'] = getenv('OPENAI_API_KEY', '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')  # Get from environment or set a default

    db.init_app(app)
    migrate.init_app(app, db)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, FileUpload


    create_database(app)
    
    #LoginManager configuration 
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    # Context processor to inject current_user into templates
    @app.context_processor
    def inject_user():
        return dict(user=current_user)
    
    # Enable CORS
    # Allow requests from React frontend (http://localhost:3000)
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

    return app

def create_database(app):
    if not path.exists(DB_NAME):
        with app.app_context():
            db.create_all()
            print('Created Database!')
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .connection import engine, Base
from ..models.student import StudentDB
from ..models.textbook_content import TextbookContentDB
from ..models.chat_session import ChatSessionDB
from ..models.message import MessageDB
from ..models.selected_text import SelectedTextDB


def init_db():
    """
    Initialize the database by creating all required tables
    """
    try:
        print("Initializing database tables...")
        
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine)
        
        print("Database tables created successfully!")
        return True
    except SQLAlchemyError as e:
        print(f"SQLAlchemy error during database initialization: {e}")
        return False
    except Exception as e:
        print(f"Error during database initialization: {e}")
        return False


def check_db_connection():
    """
    Check if the database connection is working
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return result.fetchone() is not None
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


@contextmanager
def get_db_session():
    """
    Context manager for database sessions
    """
    db = None
    try:
        db = engine.connect()
        yield db
    except Exception as e:
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    # Test database connection
    print("Checking database connection...")
    if check_db_connection():
        print("Database connection successful!")
        print("Starting database initialization...")
        if init_db():
            print("Database initialized successfully!")
        else:
            print("Database initialization failed!")
    else:
        print("Database connection failed!")
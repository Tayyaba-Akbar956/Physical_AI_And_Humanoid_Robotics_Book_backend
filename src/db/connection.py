import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("NEON_DB_URL")

# Create the SQLAlchemy engine (handle missing URL gracefully during initialization)
if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using them
        pool_size=5,         # Small pool for serverless
        max_overflow=10,     # Allow some overflow
        pool_timeout=30,      # Timeout for getting a connection
        echo=False
    )
else:
    print("Warning: NEON_DB_URL is not set. Engine will not be initialized.")
    # Create a dummy engine or handle in get_db
    engine = None

# Create a SessionLocal class to handle database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for declarative models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session for FastAPI endpoints
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Utility function to initialize the database tables
def init_db():
    """
    Create all database tables based on the defined models
    """
    print("Initializing database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise


if __name__ == "__main__":
    init_db()
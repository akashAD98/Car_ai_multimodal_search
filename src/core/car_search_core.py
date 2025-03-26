import os
import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from typing import List, Optional
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from pathlib import Path

def initialize_databases():
    """
    Initialize and connect to LanceDB databases for text and image embeddings.
    Returns tables and model classes for both databases.
    """
    # Set base directory
    base_dir = Path(__file__).parent.parent.parent
    
    # Create database folders in the db directory
    db_dir = base_dir / "db"
    os.makedirs(db_dir, exist_ok=True)
    
    text_db_folder = db_dir / "car_ai_text_embeddings"
    image_db_folder = db_dir / "car_ai_image_embeddings"
    
    os.makedirs(text_db_folder, exist_ok=True)
    os.makedirs(image_db_folder, exist_ok=True)
    
    print(f"Text database path: {text_db_folder}")
    print(f"Image database path: {image_db_folder}")
    
    # Connect to text database
    text_db = lancedb.connect(str(text_db_folder))

    # Get the embedding model for text
    text_model = get_registry().get("sentence-transformers").create(
        name="BAAI/bge-small-en-v1.5", 
        device="cpu"
    )

    # Define the car model with embeddings for text
    class CarInfo(LanceModel):
        label: str
        car_type: str
        fuel_type: str
        car_info: str = text_model.SourceField()  # Source field for embedding generation
        image_urls: List[str]
        
        # Vector field that will be automatically populated from car_info
        vector: Vector(text_model.ndims()) = text_model.VectorField()

    # Get the table for text embeddings
    text_table_name = "car_ai_text_embeddings"
    try:
        if text_table_name in text_db.table_names():
            text_table = text_db.open_table(text_table_name)
            print(f"Opened existing text table: {text_table_name}")
        else:
            # Create a new table with schema
            print(f"Creating new text table: {text_table_name}")
            
            # Create a sample data entry to initialize the table with proper schema
            sample_data = [{
                "label": "Sample Car",
                "car_type": "Sedan",
                "fuel_type": "Petrol",
                "car_info": "This is a sample car entry to initialize the database schema.",
                "image_urls": ["https://example.com/sample.jpg"]
            }]
            text_table = text_db.create_table(text_table_name, data=sample_data, schema=CarInfo)
    except Exception as e:
        print(f"Error initializing text table: {e}")
        # Fallback to create an empty table
        text_table = text_db.create_table(text_table_name, schema=CarInfo)

    # Connect to image database
    image_db = lancedb.connect(str(image_db_folder))

    # Get the embedding model for images
    image_model = get_registry().get("open-clip").create()

    # Define the Images model
    class Images(LanceModel):
        label: str
        car_info: str
        image_uri: str = image_model.SourceField()
        image_bytes: Optional[bytes] = image_model.SourceField()
        vector: Vector(image_model.ndims()) = image_model.VectorField()

    # Get the table for image embeddings
    image_table_name = "car_ai_image_embeddings"
    try:
        if image_table_name in image_db.table_names():
            image_table = image_db.open_table(image_table_name)
            print(f"Opened existing image table: {image_table_name}")
        else:
            # Create a sample entry to initialize the table
            print(f"Creating new image table: {image_table_name}")
            
            # Create a dummy 1x1 blank image
            dummy_image = Image.new('RGB', (1, 1), color='white')
            buffer = BytesIO()
            dummy_image.save(buffer, format='JPEG')
            dummy_bytes = buffer.getvalue()
            
            sample_data = [{
                "label": "Sample Car",
                "car_info": "This is a sample car image to initialize the database schema.",
                "image_uri": "https://example.com/sample.jpg",
                "image_bytes": dummy_bytes
            }]
            
            image_table = image_db.create_table(image_table_name, data=sample_data, schema=Images)
    except Exception as e:
        print(f"Error initializing image table: {e}")
        # Fallback to create an empty table
        image_table = image_db.create_table(image_table_name, schema=Images)
        
    return text_table, image_table, CarInfo, Images

def search_using_text_with_fts(text_table, query, limit=6):
    """
    Search cars using full-text search and return unique results.
    
    Args:
        text_table: LanceDB table for text search
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of unique car information dictionaries
    """
    try:
        # First try with full-text search
        results = text_table.search(query, query_type="fts").limit(limit).to_pandas()
        
        # If FTS fails or returns empty results, try vector search
        if results.empty:
            print(f"FTS search returned no results. Trying vector search for: {query}")
            results = text_table.search(query).limit(limit).to_pandas()
        
        if results.empty:
            print(f"No results found for query: {query}")
            return None

        unique_results = []
        seen_labels = set()

        for _, row in results.iterrows():
            if row["label"] not in seen_labels:
                seen_labels.add(row["label"])
                
                # Convert image_urls to a list if it's not already
                image_urls = row.get("image_urls", [])
                if isinstance(image_urls, (np.ndarray, pd.Series)):
                    image_urls = image_urls.tolist()
                elif not isinstance(image_urls, list):
                    # If it's a single string or another type, wrap it in a list
                    image_urls = [image_urls] if image_urls else []
                
                unique_results.append({
                    "label": row["label"],
                    "car_type": row.get("car_type", ""),
                    "fuel_type": row.get("fuel_type", ""),
                    "car_info": row["car_info"],
                    "image_urls": image_urls
                })

        return unique_results

    except Exception as e:
        print(f"Search Error: {e}")
        return None

def search_cars_by_image(image_table, Images, image_path, limit=6):
    """
    Search for cars using image similarity.
    
    Args:
        image_table: LanceDB table for image search
        Images: LanceDB model class for images
        image_path: Path to the query image
        limit: Maximum number of results to return
        
    Returns:
        List of Images objects representing similar cars
    """
    try:
        # Load and process the query image
        query_image = Image.open(image_path)
        results = image_table.search(query_image, vector_column_name='vector').limit(limit).to_pydantic(Images)
        
        if results:
            return results

        return None
    except Exception as e:
        print(f"Image Search Error: {e}")
        return None

def is_valid_image_path(url_or_path):
    """
    Check if a given URL or path is valid and refers to an accessible image.
    
    Args:
        url_or_path: URL or file path to check
        
    Returns:
        Boolean indicating if the path is valid
    """
    if not url_or_path:
        return False
        
    if isinstance(url_or_path, (list, np.ndarray, pd.Series)):
        if len(url_or_path) == 0:
            return False
        url_or_path = url_or_path[0]
    
    # Convert to string
    url_or_path = str(url_or_path)
    
    # Check if it's a URL
    if url_or_path.startswith(("http://", "https://")):
        try:
            # Try to make a HEAD request to check if URL exists
            response = requests.head(url_or_path, timeout=5)
            return response.status_code == 200
        except:
            return False
    else:
        # For local files, just check if the file exists
        return os.path.exists(url_or_path)

def load_image_from_url_or_path(url_or_path):
    """
    Load an image from a URL or a local path with improved error handling.
    
    Args:
        url_or_path: URL or file path to the image
        
    Returns:
        PIL.Image object or None if image could not be loaded
    """
    try:
        if not url_or_path:
            return None
            
        if isinstance(url_or_path, (list, np.ndarray, pd.Series)):
            # If it's a collection type with only one item, use that item
            if len(url_or_path) > 0:
                url_or_path = url_or_path[0]
            else:
                return None
                
        # Convert to string in case it's another type
        url_or_path = str(url_or_path)
        
        if url_or_path.startswith(("http://", "https://")):
            # Fetch image from URL
            response = requests.get(url_or_path, timeout=10)
            if response.status_code == 200:
                # Try to open the image, which might fail if content is not a valid image
                try:
                    return Image.open(BytesIO(response.content))
                except Exception:
                    # Silently fail and return None
                    return None
            else:
                # URL request failed
                return None
        else:
            # Load local image
            if os.path.exists(url_or_path):
                try:
                    return Image.open(url_or_path)
                except Exception:
                    # File exists but isn't a valid image
                    return None
            else:
                # Local file doesn't exist
                return None
    except Exception:
        # Any other errors, just return None
        return None 
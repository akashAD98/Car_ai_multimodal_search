import os
import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from PIL import Image
import io
import requests
import argparse
from pathlib import Path

def setup_database(db_uri=None, api_key=None, region=None):
    """
    Set up connection to LanceDB (local or cloud)
    
    Args:
        db_uri: URI for cloud database
        api_key: API key for cloud database
        region: Region for cloud database
        
    Returns:
        LanceDB connection
    """
    if db_uri and api_key and region:
        # Connect to LanceDB cloud
        print(f"Connecting to LanceDB cloud: {db_uri} in region {region}")
        return lancedb.connect(
            uri=db_uri,
            api_key=api_key,
            region=region
        )
    else:
        # Local LanceDB
        base_dir = Path(__file__).parent.parent.parent
        db_folder = base_dir / 'db' / 'car_ai_image_embeddings'
        # Create the directory if it doesn't exist
        os.makedirs(db_folder, exist_ok=True)
        print(f"Using local LanceDB at: {db_folder}")
        return lancedb.connect(db_folder)

def define_image_model():
    """Define the LanceDB model for images with embeddings"""
    # Get the embedding model
    func = get_registry().get("open-clip").create()
    
    # Define the Images model
    class Images(LanceModel):
        label: str
        car_info: str
        image_uri: str = func.SourceField()
        image_bytes: bytes = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()
    
    return Images, func

def process_images_from_csv(csv_path, table):
    """
    Process images from CSV and add them to LanceDB table
    
    Args:
        csv_path: Path to CSV file
        table: LanceDB table
        
    Returns:
        Number of images processed
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with columns: {df.columns.tolist()}")
    
    # Collect valid images
    image_data = []
    for _, row in df.iterrows():
        car_model = row['car_label']
        car_info = row['car_info']
        
        # Get the raw image_url string
        image_url_string = row['image_url']
        
        # Process each individual image path
        for single_path in image_url_string.split(','):
            # Clean up the path
            single_path = single_path.strip()
            
            if not single_path:
                continue
                
            try:
                # Download image if URL is provided
                if single_path.startswith("http"):
                    response = requests.get(single_path)
                    img_bytes = response.content
                else:
                    # Handle local file path
                    if os.path.exists(single_path):
                        with open(single_path, "rb") as img_file:
                            img_bytes = img_file.read()
                        
                        # Verify the image
                        with Image.open(io.BytesIO(img_bytes)) as img:
                            img.verify()
                        
                        # Add valid image to our data
                        image_data.append({
                            "label": car_model,
                            "car_info": car_info,
                            "image_uri": single_path,
                            "image_bytes": img_bytes
                        })
                        print(f"Successfully processed image: {single_path}")
                    else:
                        print(f"File not found: {single_path}")
            except Exception as e:
                print(f"Skipping corrupted image: {single_path} | Error: {e}")
    
    # Add to LanceDB
    if image_data:
        print(f"Adding {len(image_data)} valid images to the database...")
        df_images = pd.DataFrame(image_data)
        table.add(df_images)
        
        # Create full-text search index
        print("Creating full-text search index...")
        table.create_fts_index(["label", "car_info"], replace=True)
        
        return len(image_data)
    
    return 0

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Index car images in LanceDB')
    parser.add_argument('--csv', type=str, default='data/image_car_data.csv', 
                        help='Path to the CSV file containing image data')
    parser.add_argument('--cloud', action='store_true', 
                        help='Use LanceDB Cloud instead of local storage')
    parser.add_argument('--db_uri', type=str, 
                        help='LanceDB Cloud URI (required if --cloud is used)')
    parser.add_argument('--api_key', type=str, 
                        help='LanceDB Cloud API Key (required if --cloud is used)')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='LanceDB Cloud region (default: us-east-1)')
    
    args = parser.parse_args()
    
    # Validate cloud parameters
    if args.cloud and (not args.db_uri or not args.api_key):
        parser.error("--cloud requires --db_uri and --api_key")
    
    # Setup database connection
    db = setup_database(
        args.db_uri if args.cloud else None,
        args.api_key if args.cloud else None,
        args.region if args.cloud else None
    )
    
    # Get image model
    Images, _ = define_image_model()
    
    # Create or open table
    table_name = "car_ai_image_embeddings"
    if table_name in db.table_names():
        table = db.open_table(table_name)
        print(f"Opened existing table: {table_name}")
    else:
        table = db.create_table(table_name, schema=Images)
        print(f"Created new table: {table_name}")
    
    # Process images
    count = process_images_from_csv(args.csv, table)
    
    if count > 0:
        print(f"Successfully indexed {count} images!")
        
        # Test the search functionality
        search_query = "Power 86 hp car"
        try:
            print(f"Testing search with query: '{search_query}'")
            search_result = table.search(search_query, query_type="vector", vector_column_name='vector').limit(2).to_pydantic(Images)[0]
            print("Search successful! Found car:", search_result.label)
        except Exception as e:
            print(f"Search test failed: {e}")
    else:
        print("No images were indexed. Please check your CSV file and image paths.")

if __name__ == "__main__":
    main() 
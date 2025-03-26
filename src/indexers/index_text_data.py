import os
import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from typing import List
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
        db_folder = base_dir / 'db' / 'car_ai_text_embeddings'
        # Create the directory if it doesn't exist
        os.makedirs(db_folder, exist_ok=True)
        print(f"Using local LanceDB at: {db_folder}")
        return lancedb.connect(db_folder)

def define_car_model():
    """Define the LanceDB model for car info with text embeddings"""
    # Get the embedding model for text
    model = get_registry().get("sentence-transformers").create(
        name="BAAI/bge-small-en-v1.5", 
        device="cpu"
    )
    
    # Define the car model with embeddings
    class CarInfo(LanceModel):
        label: str
        car_type: str
        fuel_type: str
        car_info: str = model.SourceField()  # Source field for embedding generation
        image_urls: List[str]
        
        # Vector field that will be automatically populated from car_info
        vector: Vector(model.ndims()) = model.VectorField()
    
    return CarInfo, model

def process_data_from_csv(csv_path, table):
    """
    Process car data from CSV and add to LanceDB table
    
    Args:
        csv_path: Path to CSV file
        table: LanceDB table
        
    Returns:
        Number of entries processed
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with columns: {df.columns.tolist()}")
    
    # Prepare data for LanceDB
    processed_data = []
    for _, row in df.iterrows():
        car_model = row['car_label']
        car_info = row['car_info']
        car_type = row['car_type']
        car_fuel_type = row['fuel_type']
        
        # Get the raw image_url string and convert to list
        image_url_string = row['image_url']
        image_urls = [url.strip() for url in image_url_string.split(',') if url.strip()]
        
        if not image_urls:
            print(f"No valid image URLs for {car_model}, skipping")
            continue
            
        # Add to our processed data
        processed_data.append({
            "label": car_model,
            "car_type": car_type,
            "fuel_type": car_fuel_type,
            "car_info": car_info,
            "image_urls": image_urls
        })
    
    if processed_data:
        print(f"Adding {len(processed_data)} entries to the database...")
        df_processed = pd.DataFrame(processed_data)
        table.add(df_processed)
        
        # Create vector index for faster similarity search
        print("Creating vector index...")
        try:
            table.create_index(
                vector_column_name="vector",
                metric="cosine", 
                index_type="IVF_FLAT",  # Use cosine similarity for text embeddings
            )
            print("Vector index created successfully!")
        except Exception as e:
            print(f"Warning: Vector index creation failed: {e}")
            print("Running without vector index - search will still work but may be slower")
        
        # Create full-text search index for hybrid search capability
        print("Creating full-text search index...")
        table.create_fts_index(
            ["label", "car_info", "car_type", "fuel_type"], 
            replace=True
        )
        
        return len(processed_data)
    
    return 0

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Index car text data in LanceDB')
    parser.add_argument('--csv', type=str, default='data/text_car_data.csv', 
                        help='Path to the CSV file containing car data')
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
    
    # Get car model
    CarInfo, _ = define_car_model()
    
    # Create or open table
    table_name = "car_ai_text_embeddings"
    if table_name in db.table_names():
        table = db.open_table(table_name)
        print(f"Opened existing table: {table_name}")
    else:
        table = db.create_table(table_name, schema=CarInfo)
        print(f"Created new table: {table_name}")
    
    # Process car data
    count = process_data_from_csv(args.csv, table)
    
    if count > 0:
        print(f"Successfully indexed {count} car entries!")
        
        # Test the search functionality
        search_query = "SUV with good mileage"
        try:
            print(f"Testing search with query: '{search_query}'")
            search_result = table.search(search_query, query_type="fts").limit(1).to_pandas()
            if not search_result.empty:
                print("Search successful! Found car:", search_result['label'].iloc[0])
            else:
                print("Search returned no results")
        except Exception as e:
            print(f"Search test failed: {e}")
    else:
        print("No car data was indexed. Please check your CSV file.")

if __name__ == "__main__":
    main() 
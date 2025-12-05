"""
Build Face Embeddings Database
Creates a .pkl file with face embeddings from enrollment_database folder
Uses DeepFace with FaceNet512 model
"""

import os
import pickle
from pathlib import Path
from deepface import DeepFace
import numpy as np
from tqdm import tqdm

# Configuration
ENROLLMENT_DB_PATH = "./enrollment_database"
OUTPUT_PKL_PATH = "./face_recognition_model.pkl"
MODEL_NAME = "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace

def extract_face_embedding(image_path, model_name=MODEL_NAME):
    """
    Extract face embedding from an image using DeepFace
    
    Args:
        image_path: Path to the image file
        model_name: DeepFace model to use
        
    Returns:
        numpy array of embedding or None if face not detected
    """
    try:
        # DeepFace.represent returns a list of dictionaries with 'embedding' key
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=False,  # Don't fail if face not detected clearly
            detector_backend='opencv'  # Fast detector for enrollment
        )
        
        if embedding_objs and len(embedding_objs) > 0:
            # Return the first face's embedding
            embedding = embedding_objs[0]['embedding']
            return np.array(embedding)
        else:
            return None
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def build_embeddings_database(enrollment_path, output_path, model_name=MODEL_NAME):
    """
    Build embeddings database from enrollment folder structure
    
    Structure expected:
        enrollment_database/
            person1/
                outside.jpg
                inside.jpg
            person2/
                outside.jpg
                inside.jpg
            ...
    
    Args:
        enrollment_path: Path to enrollment_database folder
        output_path: Path to save .pkl file
        model_name: DeepFace model to use
        
    Returns:
        Dictionary with embeddings database
    """
    
    enrollment_path = Path(enrollment_path)
    
    if not enrollment_path.exists():
        raise ValueError(f"Enrollment database path does not exist: {enrollment_path}")
    
    # Database structure
    embeddings_db = {
        'model': model_name,
        'persons': {},  # person_name -> {'embeddings': [], 'image_paths': []}
        'metadata': {}
    }
    
    # Get all person folders
    person_folders = [f for f in enrollment_path.iterdir() if f.is_dir()]
    
    if len(person_folders) == 0:
        raise ValueError(f"No person folders found in {enrollment_path}")
    
    print(f"Found {len(person_folders)} persons in enrollment database")
    print(f"Using model: {model_name}")
    print("-" * 60)
    
    # Process each person
    for person_folder in tqdm(person_folders, desc="Processing persons"):
        person_name = person_folder.name
        person_embeddings = []
        person_image_paths = []
        
        # Get all images for this person
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(person_folder.glob(ext))
            image_files.extend(person_folder.glob(ext.upper()))
        
        if len(image_files) == 0:
            print(f"Warning: No images found for {person_name}")
            continue
        
        print(f"\nProcessing {person_name} ({len(image_files)} images)...")
        
        # Extract embeddings for each image
        for image_path in image_files:
            embedding = extract_face_embedding(str(image_path), model_name)
            
            if embedding is not None:
                person_embeddings.append(embedding)
                person_image_paths.append(str(image_path))
                print(f"  ✓ {image_path.name}: Embedding extracted (dim={len(embedding)})")
            else:
                print(f"  ✗ {image_path.name}: Failed to extract embedding")
        
        # Store in database if we got at least one embedding
        if len(person_embeddings) > 0:
            embeddings_db['persons'][person_name] = {
                'embeddings': person_embeddings,
                'image_paths': person_image_paths,
                'num_embeddings': len(person_embeddings)
            }
            print(f"  → Stored {len(person_embeddings)} embeddings for {person_name}")
        else:
            print(f"  → Warning: No valid embeddings for {person_name}")
    
    # Add metadata
    embeddings_db['metadata'] = {
        'num_persons': len(embeddings_db['persons']),
        'total_embeddings': sum(p['num_embeddings'] for p in embeddings_db['persons'].values()),
        'embedding_dimension': len(person_embeddings[0]) if person_embeddings else 0
    }
    
    # Save to pickle file
    print("\n" + "=" * 60)
    print(f"Saving embeddings database to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_db, f)
    
    print(f"✓ Database saved successfully!")
    print(f"  - Total persons: {embeddings_db['metadata']['num_persons']}")
    print(f"  - Total embeddings: {embeddings_db['metadata']['total_embeddings']}")
    print(f"  - Embedding dimension: {embeddings_db['metadata']['embedding_dimension']}")
    print("=" * 60)
    
    return embeddings_db

def load_embeddings_database(pkl_path):
    """Load embeddings database from pickle file"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def print_database_info(pkl_path):
    """Print information about the embeddings database"""
    db = load_embeddings_database(pkl_path)
    
    print("\n" + "=" * 60)
    print("EMBEDDINGS DATABASE INFO")
    print("=" * 60)
    print(f"Model: {db['model']}")
    print(f"Total persons: {db['metadata']['num_persons']}")
    print(f"Total embeddings: {db['metadata']['total_embeddings']}")
    print(f"Embedding dimension: {db['metadata']['embedding_dimension']}")
    print("\nPersons in database:")
    for person_name, person_data in db['persons'].items():
        print(f"  - {person_name}: {person_data['num_embeddings']} embeddings")
    print("=" * 60)

if __name__ == "__main__":
    # Build the database
    try:
        embeddings_db = build_embeddings_database(
            enrollment_path=ENROLLMENT_DB_PATH,
            output_path=OUTPUT_PKL_PATH,
            model_name=MODEL_NAME
        )
        
        # Print database info
        print_database_info(OUTPUT_PKL_PATH)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

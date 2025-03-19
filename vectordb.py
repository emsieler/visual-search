import faiss
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import json
import time
import os

from PIL import Image
from transformers import AutoModel
from models_config import model_zoo

class VectorDB:
    def __init__(self, model_names=None):
        """Load models"""
        #load all models by default
        if model_names is None:
            model_names = list(model_zoo.keys())

        self.models = self.load_models(model_names)  
        if not self.models:
            raise ValueError("No models were successfully loaded. Check dependencies!")
        
        #create dict of <key> model names and <value> ft vector dimensions
        self.vector_dims = {name: self.models[name]["embedding_dim"] for name in model_names}
        #create FAISS index for L2 (Euclidean) distance search btwn embedding vectors
        self.indexes = {name: faiss.IndexFlatL2(dim) for name, dim in self.vector_dims.items()}
        self.metadata = {}
         # Initialize lookup table for FAISS index positions
        self.image_paths = {name: {} for name in model_names}  # Maps FAISS index → image_id
        self.image_counter = 0  # total number of images in db
        self.deleted_count = 0  # Track number of deletions
        self.rebuild_threshold = 50  # Rebuild FAISS after x deletions


    def load_models(self, model_names=None):
        """Load models from models_config.py."""
        models_dict = {}

        if model_names is None:
            print("Auto-detecting available models...")
            model_names = list(model_zoo.keys())  # Load all models

        for model_name in model_names:
            if model_name in model_zoo:
                try:
                    models_dict[model_name] = model_zoo[model_name]()  # Load model
                    print(f"Loaded {model_name}")
                except Exception as e:
                    print(f"Skipping {model_name}: {e}")
            else:
                print(f"Warning: Model '{model_name}' is not recognized.")

        return models_dict

    def extract_features(self, image_path, model_name="resnet"):
        """Feature extraction"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self.models.keys())}")  
        
        # 🔹 Get the actual loaded model
        model = self.models[model_name]["model"]

        # 🔹 Get preprocessing settings from model_zoo
        model_info = model_zoo[model_name]()  # Load settings from model_zoo
        input_size = model_info["input_size"]
        mean, std = model_info["mean"], model_info["std"]

        transform = T.Compose([
            T.Resize(input_size),  
            T.ToTensor(),  
            T.Normalize(mean=mean, std=std)
        ])

        # Load model, image, and extract features
        model = self.models[model_name]["model"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = model(image_tensor).squeeze().numpy()

        return features.flatten()
    
    def add_image(self, image_path):
        """Extract embeddings for all models, add to FAISS, and update lookup table and metadata."""
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                return

            image_id = self.image_counter  # Assign unique ID

            # Extract embeddings for all models
            embeddings = {model_name: self.extract_features(image_path, model_name)
                        for model_name in self.models}

            # Store metadata (img path, etc)
            self.metadata[image_id] = {
                "path": image_path,
                "timestamp": time.time(),
                "embeddings": embeddings  # Store for reference (not for FAISS search)
            }

            # Add embeddings to FAISS and populate lookup table
            for model_name, embedding in embeddings.items():
                faiss_index = self.indexes[model_name]
                position = faiss_index.ntotal  # Get FAISS index position
                faiss_index.add(embedding.reshape(1, -1))  # Add to FAISS
                self.image_paths[model_name][position] = image_id  # Map FAISS idx → image_id
            self.image_counter += 1
            print(f"Added image {image_id}: {image_path}")

        except Exception as e:
            print(f"Error adding image {image_path}: {e}")

    def delete_entry(self, image_id):
        """Mark an entry as None for later deletion"""
        if image_id in self.metadata:
            self.metadata[image_id] = None  # Mark as deleted
            self.deleted_count += 1 
            print(f"Marked image {image_id} as deleted.")

            # Check if FAISS needs rebuilding
            if self.deleted_count >= self.rebuild_threshold:
                self.rebuild_faiss_indexes()
        else:
            print(f"Warning: Image ID {image_id} not found.")

    def rebuild_faiss_indexes(self):
        """Rebuild FAISS indexes to remove deleted entries."""
        print("Rebuilding FAISS indexes...")

        for model_name in self.indexes:
            vectors = []
            new_image_paths = {}  # New lookup table

            # Iterate over metadata and add only non-deleted embeddings
            for image_id, metadata in self.metadata.items():
                if metadata is not None and "embeddings" in metadata:
                    embedding = metadata["embeddings"].get(model_name)
                    if embedding is not None:
                        vectors.append(embedding)
                        new_image_paths[len(vectors) - 1] = image_id  # Map new FAISS index position to image_id

            # Create a new FAISS index
            dim = self.vector_dims[model_name]
            self.indexes[model_name] = faiss.IndexFlatL2(dim)
            
            if vectors:
                self.indexes[model_name].add(np.array(vectors))

            self.image_paths[model_name] = new_image_paths  # Update lookup table
            print(f"Rebuilt FAISS index for {model_name} with {len(vectors)} valid entries.")

        self.deleted_count = 0

    def search(self, query_image, model_name="resnet", k=5):
        """Search for similar images using the specified model."""
        query_vector = self.extract_features(query_image, model_name).reshape(1, -1)
        distances, indices = self.indexes[model_name].search(query_vector, k)

        results = []
        for idx in indices[0]:
            image_id = self.image_paths[model_name].get(idx)  # Map FAISS index to image ID
            if image_id is not None:
                results.append(self.metadata[image_id]["path"])
        
        return results
    
    def save(self, index_file="index.faiss", metadata_file="metadata.json", lookup_file="image_paths.json"):
        """Saves FAISS indexes, metadata, and lookup table."""
        for model_name, index in self.indexes.items():
            faiss.write_index(index, f"{model_name}_{index_file}")

        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f) #Save metadata

        with open(lookup_file, "w") as f:
            json.dump(self.image_paths, f)  #Save lookup table

    def load(self, index_file="index.faiss", metadata_file="metadata.json", lookup_file="image_paths.json"):
        """Loads FAISS indexes, metadata, and lookup table."""
        for model_name in self.models.keys():
            self.indexes[model_name] = faiss.read_index(f"{model_name}_{index_file}")

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        with open(lookup_file, "r") as f:
            self.image_paths = json.load(f)  # Restore lookup table

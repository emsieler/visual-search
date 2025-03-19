import faiss
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import json
import time
import os
import sys

from PIL import Image
from transformers import AutoModel
from models_config import model_zoo

def get_model_info(name):
    """Safely access model information from model_zoo."""
    if name not in model_zoo:
        raise ValueError(f"Model '{name}' not found in model_zoo")
    
    try:
        return model_zoo[name]()  # calls lambda and returns dict
    except Exception as e:
        raise RuntimeError(f"Error loading model '{name}': {e}")

class VectorDB:
    def __init__(self, model_names=None):
        """Initialize database and load models"""
        #load all models by default
        if model_names is None:
            model_names = list(model_zoo.keys())

        self.models = self.load_models(model_names)  
        if not self.models:
            raise ValueError("No models were successfully loaded.")
        
        #create dict of <key> model names and <value> ft vector dimensions
        self.vector_dims = {name: self.models[name]["embedding_dim"] for name in model_names}
        #create FAISS index
        self.indexes = {name: faiss.IndexFlatL2(dim) for name, dim in self.vector_dims.items()}
        self.metadata = {}
         # Init lookup table for FAISS index positions
        self.image_paths = {name: {} for name in model_names}  # Maps FAISS index → image_id
        self.image_counter = 0  
        self.deleted_count = 0 
        self.rebuild_threshold = 50 # rebuild FAISS after x deletions 
 

    def load_models(self, model_names=None):
        """Load models from models_config.py."""
        models_dict = {}

        for model_name in model_names:
            try:
                if model_name not in models_dict:
                    models_dict[model_name] = get_model_info(model_name)
                    print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Skipping {model_name}: {e}")

        return models_dict

    def extract_features(self, image_path, model_name="resnet"):
        """Feature extraction"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self.models.keys())}")  
        
        model_dict = self.models[model_name]
        
        # Special handling for CLIP model which returns a dict with model and processor
        if model_name == "clip":
            clip_dict = model_dict["model"]
            model = clip_dict["model"]
            processor = clip_dict["processor"]
            
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                # Use only the vision part of CLIP
                vision_outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
                features = vision_outputs.numpy()
                
            return features.flatten()
        else:
            # Standard handling for other models
            model = model_dict["model"]
            input_size = model_dict["input_size"]
            mean = model_dict["mean"]
            std = model_dict["std"]

            transform = T.Compose([
                T.Resize(input_size),  
                T.ToTensor(),  
                T.Normalize(mean=mean, std=std)
            ])

            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                # Handle different model output types
                if model_name in ["dino", "vit"]:  # Hugging Face models, return BaseModelOutputWithPooling object
                    output = model(image_tensor)
                    features = output.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embedding
                else:  # Standard PyTorch models ex. ResNet
                    features = model(image_tensor).squeeze().numpy()

            return features.flatten()
    
    def add_image(self, image_path):
        """Extract embeddings for all models, add to FAISS, and update lookup table and metadata."""
        try:
            # Debug prints
            print(f"Adding image: {image_path}")
            print(f"Available models: {list(self.models.keys())}")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                return

            image_id = self.image_counter 
            
            # Extract embeddings for all models
            embeddings = {}
            for model_name in self.models:
                print(f"Extracting features for {model_name}")
                embeddings[model_name] = self.extract_features(image_path, model_name)
                print(f"Features extracted for {model_name}")

            # Store metadata (img path, etc)
            self.metadata[image_id] = {
                "path": image_path,
                "timestamp": time.time(),
                "embeddings": embeddings  # Store for reference (not for FAISS search)
            }

            # Add embeddings to FAISS and populate lookup table
            for model_name, embedding in embeddings.items():
                print(f"Adding embedding to FAISS for {model_name}")
                faiss_index = self.indexes[model_name]
                position = faiss_index.ntotal  # Get FAISS index position
                faiss_index.add(embedding.reshape(1, -1))  # Add to FAISS
                self.image_paths[model_name][position] = image_id  # Map FAISS idx → image_id
            self.image_counter += 1
            print(f"Added image {image_id}: {image_path}")

        except Exception as e:
            print(f"Error adding image {image_path}: {e}")
            import traceback
            traceback.print_exc()

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

        # Convert NumPy arrays to lists for JSON serialization
        serializable_metadata = {}
        for image_id, metadata in self.metadata.items():
            if metadata is not None:
                serializable_metadata[image_id] = {
                    "path": metadata["path"],
                    "timestamp": metadata["timestamp"],
                    "embeddings": {
                        model_name: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                        for model_name, embedding in metadata["embeddings"].items()
                    }
                }
            else:
                serializable_metadata[image_id] = None

        with open(metadata_file, "w") as f:
            json.dump(serializable_metadata, f) #Save metadata

        with open(lookup_file, "w") as f:
            json.dump(self.image_paths, f)  #Save lookup table

    def load(self, index_file="index.faiss", metadata_file="metadata.json", lookup_file="image_paths.json"):
        """Loads FAISS indexes, metadata, and lookup table."""
        for model_name in self.models.keys():
            self.indexes[model_name] = faiss.read_index(f"{model_name}_{index_file}")

        with open(metadata_file, "r") as f:
            loaded_metadata = json.load(f)
            
        # Convert lists back to NumPy arrays
        self.metadata = {}
        for image_id, metadata in loaded_metadata.items():
            if metadata is not None:
                self.metadata[image_id] = {
                    "path": metadata["path"],
                    "timestamp": metadata["timestamp"],
                    "embeddings": {
                        model_name: np.array(embedding) if isinstance(embedding, list) else embedding
                        for model_name, embedding in metadata["embeddings"].items()
                    }
                }
            else:
                self.metadata[image_id] = None

        with open(lookup_file, "r") as f:
            self.image_paths = json.load(f)  # Restore lookup table

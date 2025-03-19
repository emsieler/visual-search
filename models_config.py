#define available models

import subprocess
import torch
from torchvision import models
from transformers import AutoModel, CLIPProcessor, CLIPModel, ViTModel

def load_resnet():
    """ResNet50 minus classifier for feature extraction."""
    resnet = models.resnet50(weights="IMAGENET1K_V1")
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    return resnet

def load_clip():
    """Load CLIP model and processor for vision embeddings."""
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 
    return {"model": clip_model, "processor": clip_processor, "embedding_dim": 512}

def load_dino():
    """Load DINO without the pooler layer."""
    model = ViTModel.from_pretrained("facebook/dino-vitb8", add_pooling_layer=False)
    return model

def load_vit():
    """Load ViT and remove classifier head."""
    model = AutoModel.from_pretrained("google/vit-base-patch16-224")
    return model  # No classifier, just embeddings

model_zoo = {
    "resnet": lambda: {
        "model": load_resnet(),
        "embedding_dim": 2048,
        "input_size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    #"clip": lambda: {
    #    "model": load_clip(), #stores model and preprocessor
    #    "embedding_dim": 512,
    #    "input_size": (224, 224),  # CLIP supports 336x336 as well
    #    "mean": [0.481, 0.457, 0.408],
    #    "std": [0.268, 0.261, 0.275]
    #},
    "dino": lambda: {
        "model": load_dino(),
        "embedding_dim": 768,
        "input_size": (224, 224),
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },
    "vit": lambda: {
        "model": load_vit(),
        "embedding_dim": 768,
        "input_size": (224, 224),
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },
}

def delete_cached_models():
    """Deletes cached models from Hugging Face and Torchvision."""
    cache_dirs = [
        "~/.cache/huggingface/transformers/",
        "~/.cache/torch/hub/checkpoints/"
    ]

    for cache_dir in cache_dirs:
        try:
            subprocess.run(["rm", "-rf", cache_dir], check=True)
            print(f"Deleted cache: {cache_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to delete {cache_dir}: {e}")
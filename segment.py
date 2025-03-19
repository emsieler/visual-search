import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from segment_anything import SamPredictor, sam_model_registry
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Load SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)

# Load CLIPSeg
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def apply_sam(image_path):
    """ Use Segment Anything to cut out objects. """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_point = np.array([[500, 500]])  # Manually set a point
    input_label = np.array([1])
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label)
    
    return masks[0]  # Return mask

def apply_clipseg(image_path, text_prompt="person"):
    """ Use CLIPSeg to segment based on text. """
    image = Image.open(image_path).convert("RGB")
    inputs = clipseg_processor(text=[text_prompt], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clipseg_model(**inputs)
    
    mask = outputs.logits.squeeze().sigmoid().cpu().numpy()
    return mask > 0.5  # Return binary mask

# Example Usage
image_path = "test.jpg"
sam_mask = apply_sam(image_path)
clipseg_mask = apply_clipseg(image_path, "person")

# Save masks
cv2.imwrite("sam_mask.png", sam_mask.astype(np.uint8) * 255)
cv2.imwrite("clipseg_mask.png", clipseg_mask.astype(np.uint8) * 255)

print("Segmentation done!")

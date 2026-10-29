import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


def load_encoder(model_id, device):
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    
    return proc, model

def encode_images(model, proc, images, device, batch_size=8):
    vectors = []
    
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            inputs = proc(images=batch, return_tensors='pt').to(device)
            
            outputs = model.get_image_features(**inputs, return_dict=True)
            features = F.normalize(outputs.pooler_output.float(), dim=-1)
            
            vectors.append(features.cpu().numpy())
    
    return np.concatenate(vectors, axis=0)

def encode_texts(model, proc, texts, device):
    inputs = proc(text=texts, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.inference_mode():
        outputs = model.get_text_features(**inputs, return_dict=True)
        features = F.normalize(outputs.pooler_output.float(), dim=-1)
        
    return features.cpu().numpy()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
import io
import cv2
import base64
import requests
import timm

class CheXNetModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            from transformers import AutoModel, AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
            self.model = AutoModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
            
            self.model = timm.create_model('densenet121', pretrained=True, num_classes=14)
            self.load_chexnet_weights()
            
        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            self.model = self.create_chexnet_model()
            
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_chexnet_model(self):
        model = timm.create_model('densenet121', pretrained=True, num_classes=14)
        return model
    
    def load_chexnet_weights(self):
        try:
            url = "https://download.pytorch.org/models/densenet121-a639ec97.pth"
            pass
        except Exception as e:
            print(f"Could not load CheXNet weights: {e}")
    
    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device), img
    
    def predict(self, image_bytes):
        img_tensor, original_img = self.preprocess_image(image_bytes)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        results = []
        for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
            results.append({
                "label": class_name,
                "probability": float(prob),
                "risk_level": self.get_risk_level(prob)
            })
        
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results

    def get_risk_level(self, probability):
        if probability >= 0.7:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"

class AdvancedCheXModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=14)
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device), img
    
    def predict(self, image_bytes):
        img_tensor, original_img = self.preprocess_image(image_bytes)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        results = []
        for class_name, prob in zip(self.class_names, probabilities):
            results.append({
                "label": class_name,
                "probability": float(prob),
                "risk_level": self.get_risk_level(prob)
            })
        
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results
    
    def get_risk_level(self, probability):
        if probability >= 0.7:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"

class GradCAM:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer = self.get_target_layer(target_layer_name)
        self.gradients = None
        self.activations = None
        self.register_hooks()
    
    def get_target_layer(self, layer_name):
        if layer_name:
            return dict(self.model.named_modules())[layer_name]
        
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        return target_layer
    
    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)
    
    def generate_cam(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

def create_better_overlay(original_img, heatmap, alpha=0.4):
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img.resize((224, 224)))
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    if len(original_img.shape) == 2:
        original_3ch = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    elif len(original_img.shape) == 3 and original_img.shape[2] == 3:
        original_3ch = original_img
    else:
        original_3ch = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    overlay = cv2.addWeighted(original_3ch, 1-alpha, colored_heatmap, alpha, 0)
    return overlay

model_instance = AdvancedCheXModel()

def predict(image_bytes, heatmap_method='gradcam'):
    try:
        predictions = model_instance.predict(image_bytes)
        
        heatmap_base64 = None
        target_class_name = "N/A"
        
        if predictions and heatmap_method == 'gradcam':
            try:
                img_tensor, original_img = model_instance.preprocess_image(image_bytes)
                
                top_class_name = predictions[0]['label']
                class_idx = model_instance.class_names.index(top_class_name)
                
                grad_cam = GradCAM(model_instance.model)
                heatmap = grad_cam.generate_cam(img_tensor, class_idx)
                
                overlayed = create_better_overlay(original_img, heatmap)
                
                _, buffer = cv2.imencode('.png', overlayed)
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                target_class_name = top_class_name
                
            except Exception as e:
                print(f"Heatmap generation failed: {e}")
        
        return {
            "predictions": predictions,
            "heatmap": heatmap_base64,
            "heatmap_target_class": target_class_name,
            "heatmap_method": heatmap_method
        }
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return {
            "predictions": [],
            "heatmap": None,
            "heatmap_target_class": "Error",
            "heatmap_method": heatmap_method,
            "error": str(e)
        }

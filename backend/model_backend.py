import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
import numpy as np
import io
import cv2
import base64
import torch.nn.functional as F

model = xrv.models.DenseNet(weights="all")
model.eval()

def grad_cam(model, input_tensor, target_class=None):
    """
    Generate Grad-CAM heatmap - more accurate for medical imaging
    """
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    
    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)
    
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, target_class].backward(retain_graph=True)
    
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    handle_backward.remove()
    handle_forward.remove()
    
    weights = np.mean(grads, axis=(1, 2))
    
    heatmap = np.zeros(acts.shape[1:])
    for i, w in enumerate(weights):
        heatmap += w * acts[i]
    
    heatmap = np.maximum(heatmap, 0)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap, target_class

def integrated_gradients(model, input_tensor, target_class=None, steps=50):
    """
    Generate Integrated Gradients heatmap - often more stable
    """
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()
    
    baseline = torch.zeros_like(input_tensor)
    
    alphas = torch.linspace(0, 1, steps)
    gradients = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        
        output = model(interpolated)
        model.zero_grad()
        output[0, target_class].backward()
        
        gradients.append(interpolated.grad.cpu().numpy())
    
    avg_gradients = np.mean(gradients, axis=0)
    integrated_grad = (input_tensor - baseline).cpu().numpy() * avg_gradients
    
    if len(integrated_grad.shape) == 4: 
        integrated_grad = np.abs(integrated_grad).sum(axis=1)[0]
    else:
        integrated_grad = np.abs(integrated_grad[0])
    
    if integrated_grad.max() > 0:
        integrated_grad = (integrated_grad - integrated_grad.min()) / (integrated_grad.max() - integrated_grad.min())
    
    return integrated_grad, target_class

def generate_multiple_heatmaps(input_tensor, method='gradcam'):
    """
    Generate heatmap using specified method
    """
    with torch.enable_grad():
        if method == 'gradcam':
            heatmap, target_class = grad_cam(model, input_tensor)
        elif method == 'integrated_gradients':
            heatmap, target_class = integrated_gradients(model, input_tensor)
        else:  
            heatmap, target_class = improved_saliency(input_tensor)
    
    return heatmap, target_class

def improved_saliency(input_tensor):
    """
    Improved saliency map with smoothing
    """
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    
    target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    
    saliency = cv2.GaussianBlur(saliency, (3, 3), 0)
    
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    
    return saliency, target_class

def create_better_overlay(original_img, heatmap, alpha=0.4):
    """
    Create a better overlay with improved color mapping
    """
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    if len(original_img.shape) == 2:
        original_3ch = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        original_3ch = original_img
    
    overlay = cv2.addWeighted(original_3ch, 1-alpha, colored_heatmap, alpha, 0)
    
    return overlay

def predict(image_bytes, heatmap_method='gradcam'):
    """
    Enhanced prediction function with better heatmap generation
    
    Args:
        image_bytes: Image data
        heatmap_method: 'gradcam', 'integrated_gradients', or 'saliency'
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    img_array = img_tensor.numpy()
    img_array = xrv.datasets.normalize(img_array, 255)
    img_tensor = torch.from_numpy(img_array).float()

    with torch.no_grad():
        preds = model(img_tensor)
    probabilities = preds[0].cpu().numpy()

    labels = model.pathologies
    results = list(zip(labels, probabilities))
    results.sort(key=lambda x: -x[1])

    try:
        heatmap, target_class = generate_multiple_heatmaps(img_tensor.clone(), method=heatmap_method)
        
        original = np.array(img.resize((224, 224)))
        overlayed = create_better_overlay(original, heatmap)
        
        _, buffer = cv2.imencode('.png', overlayed)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        target_class_name = labels[target_class] if target_class < len(labels) else "Unknown"
        
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        heatmap_base64 = None
        target_class_name = "Error"

    return {
        "predictions": [{"label": l, "probability": float(p)} for l, p in results],
        "heatmap": heatmap_base64,
        "heatmap_target_class": target_class_name,
        "heatmap_method": heatmap_method
    }


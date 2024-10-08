import torch
from torchvision import models
import torch.nn as nn
import json
from PIL import Image
import numpy as np
import argparse

# Define the arguments to be passed via the command line
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

# Load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_features = 25088
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_features = model.fc.in_features
    else:
        raise ValueError("Unsupported model architecture in checkpoint")
    
    # Build classifier
    classifier = nn.Sequential(nn.Linear(input_features, 512),
                               nn.ReLU(),
                               nn.Linear(512, 102),
                               nn.LogSoftmax(dim=1))
    
    if checkpoint['arch'].startswith('vgg'):
        model.classifier = classifier
    elif checkpoint['arch'].startswith('resnet'):
        model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Process image
def process_image(image_path):
    img = Image.open(image_path)
    
    # Process image: resize, crop, normalize
    img.thumbnail((256, 256))
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.tensor(np_image)

# Prediction function
def predict(image_path, model, top_k, device):
    model.to(device)
    img = process_image(image_path)
    img = img.unsqueeze_(0).float().to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(img)
        probs, class_idxs = torch.exp(output).topk(top_k, dim=1)
    
    # Convert indices to actual class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in class_idxs[0].cpu().numpy()]
    
    return probs[0].tolist(), classes

# Load category names
def load_category_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint)
    
    probs, classes = predict(args.input, model, args.top_k, device)
    
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        classes = [cat_to_name[str(c)] for c in classes]
    
    print(f"Predicted classes: {classes}")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()

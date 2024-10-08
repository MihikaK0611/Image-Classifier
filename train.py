import torch
import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict

# Define the arguments to be passed via the command line
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network model on flower images")
    
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for model')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    return parser.parse_args()

# Define transforms for training and validation sets
def get_data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define your transforms for the training and validation sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return train_loader, valid_loader, train_data

# Create and configure the model
def create_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_features = 25088  # Input size for VGG models
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_features = model.fc.in_features  # Input size for ResNet models
    else:
        raise ValueError("Unsupported architecture")
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),  # 102 is the number of flower classes
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    if arch.startswith('vgg'):
        model.classifier = classifier
    elif arch.startswith('resnet'):
        model.fc = classifier
    
    return model

# Train and validate the model
def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs):
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} running...")
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        valid_loss, accuracy = validate_model(model, valid_loader, device, criterion)
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

# Validation function
def validate_model(model, valid_loader, device, criterion):
    model.eval()
    valid_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            valid_loss += criterion(logps, labels).item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return valid_loss, accuracy

# Save the checkpoint
def save_checkpoint(model, train_data, save_dir, arch):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': arch,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, save_dir)

def main():
    args = get_input_args()
    train_loader, valid_loader, train_data = get_data_loaders(args.data_dir)
    
    model = create_model(args.arch, args.hidden_units)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    train_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs)
    
    save_checkpoint(model, train_data, args.save_dir, args.arch)

if __name__ == "__main__":
    main()

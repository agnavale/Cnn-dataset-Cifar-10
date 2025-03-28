import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# Define CIFAR-10 classes
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce size to 16x16

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce size to 8x8
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
# model :cifar10_cnn, 80% accuracy
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device)) 
model.to(device)
model.eval()


# Define transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Prediction function
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    category = CIFAR10_CLASSES[predicted.item()]  
    return category



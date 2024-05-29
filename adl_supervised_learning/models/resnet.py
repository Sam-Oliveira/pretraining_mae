import torch
import torch.nn as nn
import torchvision.models as models

class ResNetSegmentation(nn.Module):
    def __init__(self, n_classes):
        super(ResNetSegmentation, self).__init__()
        
        # Load a pre-trained ResNet
        self.base_model = models.resnet50(pretrained=False)
        
        # Remove the fully connected layer of the ResNet
        self.base_layers = nn.Sequential(*list(self.base_model.children())[:-2])
        
        # Adjust upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to 14x14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # Upsample to 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, n_classes, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample to 224x224
        )

    def forward(self, x):
        x = self.base_layers(x)
        x = self.upsample(x)
        return x
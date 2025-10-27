# Emotion Recognition CNN

A convolutional neural network (CNN) built with **PyTorch** for **facial emotion recognition**.  
This project was developed as part of a computer vision course to classify facial expressions into eight distinct emotions.


## Overview
The model takes an RGB image of a face as input and predicts one of eight emotion classes (e.g., happy, sad, angry, surprised, etc.).  
It uses a simple yet effective CNN architecture trained on a facial expression dataset.

### Architecture
- **3 convolutional layers** with ReLU activation and max pooling  
- **1 fully connected hidden layer** (512 units)  
- **Dropout (0.25)** for regularization  
- **8-way softmax output layer**

```python
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

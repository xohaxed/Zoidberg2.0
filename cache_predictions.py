"""
Fixed Cache Predictions Script
===============================
Loads raw images directly (like notebook 04) for proper normalization
"""

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from tqdm import tqdm
import json
import cv2
from sklearn.model_selection import train_test_split

print("="*60)
print("CACHING PREDICTIONS (FIXED - Using Raw Images)")
print("="*60)

# Setup
project_root = Path.cwd()
device = torch.device('cpu')

# ==================== Model Architectures ====================

class PneumoniaCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.3):
        super(PneumoniaCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG16Transfer(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True):
        super(VGG16Transfer, self).__init__()
        
        vgg16 = models.vgg16(weights=None)
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True):
        super(ResNet50Transfer, self).__init__()
        
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_layers:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================== Load Models ====================

print("\n[1/4] Loading models...")
loaded_models = {}
models_dir = project_root / 'models' / 'saved_models'

# Load Deep Learning models
dl_models_dir = models_dir / 'deep_learning'
if dl_models_dir.exists():
    # Custom CNN
    custom_cnn_path = dl_models_dir / 'custom_cnn.pth'
    if custom_cnn_path.exists():
        model = PneumoniaCNN(dropout_rate=0.3)
        checkpoint = torch.load(custom_cnn_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()
        loaded_models['Custom CNN'] = {'model': model, 'type': 'deep_learning'}
        print(f"  ✓ Custom CNN")
    
    # VGG16
    vgg16_path = dl_models_dir / 'vgg16_transfer.pth'
    if vgg16_path.exists():
        model = VGG16Transfer(num_classes=2, freeze_layers=True)
        checkpoint = torch.load(vgg16_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()
        loaded_models['VGG16 Transfer'] = {'model': model, 'type': 'deep_learning'}
        print(f"  ✓ VGG16 Transfer")
    
    # ResNet50
    resnet50_path = dl_models_dir / 'resnet50_transfer.pth'
    if resnet50_path.exists():
        model = ResNet50Transfer(num_classes=2, freeze_layers=True)
        checkpoint = torch.load(resnet50_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()
        loaded_models['ResNet50 Transfer'] = {'model': model, 'type': 'deep_learning'}
        print(f"  ✓ ResNet50 Transfer")

# Load Baseline ML models
baseline_models = ['gradient_boosting', 'random_forest', 'svm_rbf', 'logistic_regression']
for model_name in baseline_models:
    model_path = models_dir / f'baseline_{model_name}.pkl'
    if model_path.exists():
        model = joblib.load(model_path)
        display_name = model_name.replace('_', ' ').title()
        loaded_models[display_name] = {'model': model, 'type': 'baseline'}
        print(f"  ✓ {display_name}")

print(f"\nLoaded {len(loaded_models)} models")

# ==================== Load Test Data (RAW IMAGES - Like Notebook 04) ====================

print("\n[2/4] Loading test data (raw images with proper normalization)...")

# Load PCA features for baseline models
data_dir = project_root / 'data' / 'processed'
X_test_pca = np.load(data_dir / 'X_test.npy')

# Load preprocessing metadata
with open(data_dir / 'preprocessing_metadata.json', 'r') as f:
    preprocessing_meta = json.load(f)
target_size = tuple(preprocessing_meta['target_size'])

# Get all image paths from raw data (SAME AS NOTEBOOK 04)
data_raw = project_root / 'data' / 'raw'
all_image_paths = []
all_labels = []

# Dataset1
dataset1_dir = data_raw / 'dataset1'
for class_name in ['NORMAL', 'PNEUMONIA']:
    class_dir = dataset1_dir / class_name
    if class_dir.exists():
        image_files = list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        all_image_paths.extend([str(p) for p in image_files])
        all_labels.extend([class_name] * len(image_files))

# Dataset2
dataset2_dir = data_raw / 'dataset2'
for class_name in ['NORMAL', 'PNEUMONIA']:
    class_dir = dataset2_dir / class_name
    if class_dir.exists():
        image_files = list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        all_image_paths.extend([str(p) for p in image_files])
        all_labels.extend([class_name] * len(image_files))

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

# Encode labels
label_encoder_path = models_dir / 'label_encoder.pkl'
if label_encoder_path.exists():
    label_encoder = joblib.load(label_encoder_path)
    all_labels_encoded = label_encoder.transform(all_labels)
else:
    all_labels_encoded = np.array([0 if l == 'NORMAL' else 1 for l in all_labels])

# Split to get test set (SAME SPLIT AS NOTEBOOK 04 - seed=42)
_, X_test_paths, _, y_test = train_test_split(
    all_image_paths, all_labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=all_labels_encoded
)

print(f"  Found {len(X_test_paths)} test images")

# Compute dataset normalization statistics from training set (SAME AS NOTEBOOK 04)
print("  Computing normalization statistics...")
X_train_paths, _, _, _ = train_test_split(
    all_image_paths, all_labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=all_labels_encoded
)

# Sample for statistics
sample_size = min(1000, len(X_train_paths))
sample_paths = np.random.choice(X_train_paths, sample_size, replace=False)
pixel_values = []

for img_path in tqdm(sample_paths, desc="  Sampling for stats"):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, target_size)
        pixel_values.append(img.flatten())

all_pixels = np.concatenate(pixel_values)
dataset_mean = all_pixels.mean() / 255.0
dataset_std = all_pixels.std() / 255.0

print(f"  ✓ Dataset statistics: mean={dataset_mean:.4f}, std={dataset_std:.4f}")

# Load test images with PROPER NORMALIZATION (SAME AS NOTEBOOK 04)
print(f"  Loading {len(X_test_paths)} test images with normalization...")
X_test_images = []

for img_path in tqdm(X_test_paths, desc="  Loading test images"):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Warning: Failed to load {img_path}")
        continue
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Standardize using dataset statistics (CRITICAL FOR PERFORMANCE)
    img = (img - dataset_mean) / (dataset_std + 1e-7)
    
    # Add channel dimension (1, H, W)
    img = np.expand_dims(img, axis=0)
    X_test_images.append(img)

X_test_images = np.array(X_test_images)

print(f"\n  ✓ Test images: {X_test_images.shape}")
print(f"  ✓ Test PCA features: {X_test_pca.shape}")
print(f"  ✓ Test labels: {y_test.shape}")
print(f"  ✓ Label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ==================== Generate Predictions ====================

print("\n[3/4] Generating predictions...")
all_predictions = {}

for model_name, model_info in tqdm(loaded_models.items(), desc="Models"):
    model = model_info['model']
    model_type = model_info['type']
    
    try:
        if model_type == 'deep_learning':
            model.eval()
            all_probs = []
            
            with torch.no_grad():
                for i in range(0, len(X_test_images), 32):
                    batch = X_test_images[i:i+32]
                    batch_tensor = torch.FloatTensor(batch).to(device)
                    outputs = model(batch_tensor)
                    probs = F.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
            
            all_probs = np.array(all_probs)
            pred_labels = np.argmax(all_probs, axis=1)
            pred_probs = all_probs[:, 1]
            
        else:  # baseline
            pred_labels = model.predict(X_test_pca)
            if hasattr(model, 'predict_proba'):
                pred_probs = model.predict_proba(X_test_pca)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(X_test_pca)
                pred_probs = 1 / (1 + np.exp(-decision))
            else:
                pred_probs = pred_labels.astype(float)
        
        all_predictions[model_name] = {
            'labels': pred_labels,
            'probs': pred_probs,
            'type': model_type
        }
        
    except Exception as e:
        print(f"  ❌ Error with {model_name}: {str(e)}")

print(f"\n  ✓ Generated predictions for {len(all_predictions)} models")

# ==================== Save Cache ====================

print("\n[4/4] Saving cache...")
cache_path = project_root / 'reports' / 'metrics' / 'cached_predictions.npz'
cache_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(cache_path, predictions=all_predictions, y_test=y_test)
print(f"  ✓ Saved to: {cache_path}")

# Save summary with metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

summary = {
    'num_models': len(all_predictions),
    'num_samples': len(y_test),
    'models': {},
    'normalization': {
        'mean': float(dataset_mean),
        'std': float(dataset_std),
        'target_size': target_size
    }
}

for model_name, pred in all_predictions.items():
    try:
        summary['models'][model_name] = {
            'roc_auc': float(roc_auc_score(y_test, pred['probs'])),
            'accuracy': float(accuracy_score(y_test, pred['labels'])),
            'precision': float(precision_score(y_test, pred['labels'])),
            'recall': float(recall_score(y_test, pred['labels'])),
            'f1_score': float(f1_score(y_test, pred['labels'])),
            'type': pred['type']
        }
    except:
        pass

summary_path = project_root / 'reports' / 'metrics' / 'cache_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Summary saved to: {summary_path}")

print("\n" + "="*60)
print("✅ CACHE COMPLETE WITH PROPER NORMALIZATION!")
print("="*60)
print("\nPerformance Preview:")
for model_name in sorted(summary['models'].keys(), key=lambda x: summary['models'][x]['roc_auc'], reverse=True):
    metrics = summary['models'][model_name]
    print(f"  {model_name:25s} - ROC-AUC: {metrics['roc_auc']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

print("\nDashboard will now show correct metrics!")
print("Run: streamlit run dashboard.py")

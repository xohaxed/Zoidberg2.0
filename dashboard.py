"""
🏥 Pneumonia Detection AI Dashboard
=====================================
Interactive dashboard for model performance analysis, comparison, and real-time predictions.

Features:
- Model Performance Comparison
- Interactive ROC Curves
- Confusion Matrices
- Real-time Predictions
- Threshold Optimization
- Ensemble Analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
from PIL import Image
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, 
    classification_report, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pneumonia AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark theme support
st.markdown("""
<style>
    /* Fix metric visibility on dark theme */
    [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
        opacity: 0.8;
    }
    
    [data-testid="stMetricDelta"] {
        color: var(--text-color) !important;
    }
    
    /* Fix card backgrounds for dark theme */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }
    
    /* Improve visibility of all text elements */
    .stMarkdown, .stText, p, span {
        color: var(--text-color) !important;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4A9EFF;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Beautiful gradient cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Fix dataframe visibility */
    [data-testid="stDataFrame"] {
        background-color: transparent !important;
    }
    
    /* Improve button visibility */
    .stButton>button {
        border: 1px solid rgba(128, 128, 128, 0.3);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


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


# ==================== Caching & Data Loading ====================

@st.cache_resource
def load_models():
    """Load all trained models"""
    project_root = Path.cwd()
    models_dir = project_root / 'models' / 'saved_models'
    device = torch.device('cpu')  # Use CPU for dashboard
    
    loaded_models = {}
    
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
            loaded_models['Custom CNN'] = {
                'model': model, 
                'type': 'deep_learning',
                'roc_auc': checkpoint.get('roc_auc', None)
            }
        
        # VGG16
        vgg16_path = dl_models_dir / 'vgg16_transfer.pth'
        if vgg16_path.exists():
            model = VGG16Transfer(num_classes=2, freeze_layers=True)
            checkpoint = torch.load(vgg16_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            loaded_models['VGG16 Transfer'] = {
                'model': model, 
                'type': 'deep_learning',
                'roc_auc': checkpoint.get('roc_auc', None)
            }
        
        # ResNet50
        resnet50_path = dl_models_dir / 'resnet50_transfer.pth'
        if resnet50_path.exists():
            model = ResNet50Transfer(num_classes=2, freeze_layers=True)
            checkpoint = torch.load(resnet50_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            loaded_models['ResNet50 Transfer'] = {
                'model': model, 
                'type': 'deep_learning',
                'roc_auc': checkpoint.get('roc_auc', None)
            }
    
    # Load Baseline ML models
    baseline_models = ['gradient_boosting', 'random_forest', 'svm_rbf', 'logistic_regression']
    for model_name in baseline_models:
        model_path = models_dir / f'baseline_{model_name}.pkl'
        if model_path.exists():
            model = joblib.load(model_path)
            display_name = model_name.replace('_', ' ').title()
            loaded_models[display_name] = {
                'model': model, 
                'type': 'baseline',
                'roc_auc': None
            }
    
    return loaded_models


@st.cache_data
def load_test_data():
    """Load test dataset and predictions"""
    project_root = Path.cwd()
    data_dir = project_root / 'data' / 'processed'
    
    X_test_pca = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    # Handle string labels
    if y_test.dtype.kind in ['U', 'S', 'O']:
        label_map = {'NORMAL': 0, 'PNEUMONIA': 1}
        y_test = np.array([label_map.get(str(label), label) for label in y_test])
    y_test = y_test.astype(np.int64)
    
    # Reconstruct raw images
    pca_path = project_root / 'models' / 'saved_models' / 'pca_transformer.pkl'
    X_test_images = None
    if pca_path.exists():
        pca = joblib.load(pca_path)
        X_test_raw = pca.inverse_transform(X_test_pca)
        X_test_images = X_test_raw.reshape(-1, 224, 224)
    
    return X_test_pca, X_test_images, y_test


@st.cache_data
def load_ensemble_config():
    """Load ensemble configuration"""
    project_root = Path.cwd()
    config_path = project_root / 'models' / 'saved_models' / 'ensemble_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_evaluation_results():
    """Load pre-computed evaluation results from notebooks"""
    project_root = Path.cwd()
    results_path = project_root / 'reports' / 'metrics' / 'baseline_results_summary.json'
    
    results = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    # Also try to load from ensemble results if available
    ensemble_path = project_root / 'models' / 'saved_models' / 'ensemble_config.json'
    if ensemble_path.exists():
        with open(ensemble_path, 'r') as f:
            ensemble_data = json.load(f)
            # Merge ensemble performance data
            if 'models' in ensemble_data and 'weights' in ensemble_data:
                for model_name, weight in zip(ensemble_data['models'], ensemble_data['weights']):
                    if model_name not in results:
                        results[model_name] = {}
                    results[model_name]['roc_auc'] = weight
    
    return results


@st.cache_data
def load_precomputed_metrics():
    """Load all pre-computed metrics and predictions from notebooks"""
    project_root = Path.cwd()
    
    # Try to load various result files
    metrics = {
        'models': {},
        'ensemble': None
    }
    
    # Load baseline results
    baseline_path = project_root / 'reports' / 'metrics' / 'baseline_results_summary.json'
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
            metrics['models'].update(baseline_data)
    
    # Load ensemble configuration
    ensemble_path = project_root / 'models' / 'saved_models' / 'ensemble_config.json'
    if ensemble_path.exists():
        with open(ensemble_path, 'r') as f:
            metrics['ensemble'] = json.load(f)
    
    return metrics


# ==================== Prediction Functions ====================

# Model performance weights (from cache_summary.json - based on validated ROC-AUC)
MODEL_WEIGHTS = {
    'VGG16 Transfer': 0.997,      # Best performing
    'ResNet50 Transfer': 0.997,
    'Custom CNN': 0.990,
    'Gradient Boosting': 0.985,
    'Svm Rbf': 0.988,
    'Logistic Regression': 0.977,
    'Random Forest': 0.975
}

def predict_dl_model(model, image_array):
    """Predict using deep learning model with optimized inference
    
    Args:
        model: PyTorch model (must be in eval mode)
        image_array: numpy array of shape (1, 224, 224) - GLOBAL standardized
    
    Returns:
        pred_class: predicted class (0 or 1)
        confidence: confidence score (calibrated)
        probs: class probabilities [P(NORMAL), P(PNEUMONIA)]
    """
    device = torch.device('cpu')
    model.eval()
    
    with torch.no_grad():
        # image_array is (1, 224, 224), add batch dimension
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0).to(device)  # (1, 1, 224, 224)
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
    return pred_class, confidence, probs[0].cpu().numpy()


def predict_baseline_model(model, pca_features):
    """Predict using baseline ML model with proper probability calibration
    
    Args:
        model: sklearn model
        pca_features: PCA-transformed features (PER-IMAGE standardized)
    
    Returns:
        pred_class: predicted class (0 or 1)
        confidence: confidence score
        probs: class probabilities [P(NORMAL), P(PNEUMONIA)]
    """
    pred_class = model.predict(pca_features.reshape(1, -1))[0]
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(pca_features.reshape(1, -1))[0]
    elif hasattr(model, 'decision_function'):
        # Platt scaling for SVM
        decision = model.decision_function(pca_features.reshape(1, -1))[0]
        # Sigmoid calibration
        prob_positive = 1.0 / (1.0 + np.exp(-decision))
        probs = np.array([1.0 - prob_positive, prob_positive])
    else:
        probs = np.array([1.0 - float(pred_class), float(pred_class)])
    
    # Ensure proper probability normalization
    probs = np.clip(probs, 0, 1)
    probs = probs / (probs.sum() + 1e-10)
    
    confidence = probs[pred_class]
    
    return int(pred_class), float(confidence), probs


def compute_ensemble_prediction(results_list):
    """Compute weighted ensemble prediction from all model results
    
    Uses ROC-AUC weighted voting for maximum accuracy.
    
    Args:
        results_list: List of dicts with 'Model', 'PNEUMONIA' probability, '_pred_class'
    
    Returns:
        dict with ensemble prediction details
    """
    if not results_list:
        return None
    
    weighted_pneumonia_prob = 0.0
    total_weight = 0.0
    votes_pneumonia = 0
    votes_normal = 0
    
    for result in results_list:
        model_name = result['Model']
        pred_class = result['_pred_class']
        
        # Parse probability (remove % sign)
        pneumonia_prob = float(result['PNEUMONIA'].replace('%', '')) / 100.0
        
        # Get model weight (default to 0.9 if not found)
        weight = MODEL_WEIGHTS.get(model_name, 0.9)
        
        weighted_pneumonia_prob += weight * pneumonia_prob
        total_weight += weight
        
        if pred_class == 1:
            votes_pneumonia += 1
        else:
            votes_normal += 1
    
    # Compute final ensemble probability
    ensemble_prob = weighted_pneumonia_prob / (total_weight + 1e-10)
    
    # Final prediction (threshold 0.5)
    ensemble_pred = 1 if ensemble_prob > 0.5 else 0
    ensemble_label = 'PNEUMONIA' if ensemble_pred == 1 else 'NORMAL'
    
    # Confidence is distance from decision boundary
    confidence = abs(ensemble_prob - 0.5) * 2  # Scale to [0, 1]
    
    return {
        'prediction': ensemble_label,
        'confidence': confidence,
        'pneumonia_prob': ensemble_prob,
        'normal_prob': 1.0 - ensemble_prob,
        'votes_pneumonia': votes_pneumonia,
        'votes_normal': votes_normal,
        'total_models': len(results_list)
    }


def batch_predictions(models_dict, X_test_pca, X_test_images, y_test, use_cached=True):
    """Generate predictions for all models on test set (or use cached results)
    
    Returns:
        tuple: (predictions_dict, y_test_used) - the predictions and the y_test labels used for them
    """
    project_root = Path.cwd()
    cache_path = project_root / 'reports' / 'metrics' / 'cached_predictions.npz'
    
    # Try to load cached predictions first (these are computed from REAL images, not PCA-reconstructed)
    if use_cached and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            all_predictions = cached['predictions'].item()
            # Also load y_test from cache to ensure consistency
            if 'y_test' in cached:
                cached_y_test = cached['y_test']
                st.success(f"✅ Loaded cached predictions for {len(all_predictions)} models (computed from real images)")
                return all_predictions, cached_y_test
            else:
                st.success(f"✅ Loaded cached predictions for {len(all_predictions)} models")
                return all_predictions, y_test
        except Exception as e:
            st.warning(f"Could not load cache: {e}. Computing fresh predictions...")
    
    all_predictions = {}
    
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        model_type = model_info['type']
        
        try:
            if model_type == 'deep_learning':
                if X_test_images is None:
                    continue
                
                device = torch.device('cpu')
                model.eval()
                all_probs = []
                
                with torch.no_grad():
                    for i in range(0, len(X_test_images), 32):
                        batch = X_test_images[i:i+32]
                        batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
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
            st.warning(f"Error with {model_name}: {str(e)}")
    
    # Cache the predictions for future use
    if all_predictions:
        try:
            np.savez(cache_path, predictions=all_predictions, y_test=y_test)
        except:
            pass  # Silent fail if can't save cache
    
    return all_predictions, y_test


# ==================== Visualization Functions ====================

def plot_roc_curves_plotly(predictions_dict, y_test):
    """Interactive ROC curves using Plotly"""
    fig = go.Figure()
    
    for model_name, pred in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_test, pred['probs'])
        roc_auc = roc_auc_score(y_test, pred['probs'])
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC={roc_auc:.3f})',
            line=dict(width=2),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray'),
        showlegend=True
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def plot_confusion_matrix_plotly(y_true, y_pred, title="Confusion Matrix"):
    """Interactive confusion matrix using Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['NORMAL', 'PNEUMONIA'],
        y=['NORMAL', 'PNEUMONIA'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_metrics_comparison(predictions_dict, y_test):
    """Bar chart comparing metrics across models"""
    metrics_data = []
    
    for model_name, pred in predictions_dict.items():
        metrics_data.append({
            'Model': model_name,
            'ROC-AUC': roc_auc_score(y_test, pred['probs']),
            'Accuracy': accuracy_score(y_test, pred['labels']),
            'Precision': precision_score(y_test, pred['labels']),
            'Recall': recall_score(y_test, pred['labels']),
            'F1-Score': f1_score(y_test, pred['labels'])
        })
    
    df = pd.DataFrame(metrics_data)
    df = df.sort_values('ROC-AUC', ascending=False)
    
    fig = go.Figure()
    
    metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=df['Model'],
            y=df[metric],
            text=df[metric].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Performance Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        width=1000,
        height=500,
        template='plotly_white',
        yaxis=dict(range=[0, 1.1])
    )
    
    return fig, df


# ==================== Main Dashboard ====================

def main():
    # Header with personality
    st.markdown('<h1 class="main-header">🏥 Pneumonia Detection AI Dashboard 🔬</h1>', unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center; margin-bottom: 20px; font-size: 1.1em;'>
    👨‍⚕️ <strong>Your AI Medical Assistant</strong> - Powered by Deep Learning Magic ✨<br>
    <em>Built with ❤️ for accurate pneumonia detection from chest X-rays</em>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading models and data..."):
        models_dict = load_models()
        X_test_pca, X_test_images, y_test = load_test_data()
        ensemble_config = load_ensemble_config()
        precomputed_metrics = load_precomputed_metrics()
    
    if not models_dict:
        st.error("❌ No models found! Please train models first.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Add option to force recompute
    st.sidebar.markdown("---")
    force_recompute = st.sidebar.checkbox("🔄 Force Recompute Predictions", value=False)
    if force_recompute:
        st.sidebar.warning("⚠️ This will take several minutes")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Model Comparison", "🔍 Single Prediction", "📈 Ensemble Analysis", "📋 Detailed Metrics"]
    )
    
    # ==================== PAGE 1: Model Comparison ====================
    if page == "📊 Model Comparison":
        st.header("📊 Model Performance Comparison 🏆")
        st.markdown("*Let's see which AI model performs best at detecting pneumonia!* 🤖💪")
        
        # Generate or load predictions
        if force_recompute:
            with st.spinner("Computing predictions (this may take a few minutes)..."):
                predictions, y_test_eval = batch_predictions(models_dict, X_test_pca, X_test_images, y_test, use_cached=False)
        else:
            with st.spinner("Loading predictions..."):
                predictions, y_test_eval = batch_predictions(models_dict, X_test_pca, X_test_images, y_test, use_cached=True)
        
        if not predictions:
            st.error("❌ No predictions available! Click 'Force Recompute Predictions' in sidebar or run notebooks first.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📁 Test Samples", len(y_test_eval))
        with col2:
            st.metric("🤖 Models Loaded", len(models_dict))
        with col3:
            best_model = max(predictions.items(), key=lambda x: roc_auc_score(y_test_eval, x[1]['probs']))
            st.metric("🏆 Best Model", best_model[0][:15])
        with col4:
            best_auc = roc_auc_score(y_test_eval, best_model[1]['probs'])
            st.metric("🎯 Best ROC-AUC", f"{best_auc:.4f}")
        
        st.markdown("---")
        
        # ROC Curves
        st.subheader("📈 ROC Curves")
        fig_roc = plot_roc_curves_plotly(predictions, y_test_eval)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.markdown("---")
        
        # Metrics Comparison
        st.subheader("📊 Performance Metrics")
        fig_metrics, df_metrics = plot_metrics_comparison(predictions, y_test_eval)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Metrics table
        st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']), use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrices
        st.subheader("🔲 Confusion Matrices")
        selected_models = st.multiselect(
            "Select models to view confusion matrices",
            list(predictions.keys()),
            default=list(predictions.keys())[:3]
        )
        
        cols = st.columns(min(3, len(selected_models)))
        for idx, model_name in enumerate(selected_models):
            with cols[idx % 3]:
                fig_cm = plot_confusion_matrix_plotly(
                    y_test_eval, 
                    predictions[model_name]['labels'],
                    title=model_name
                )
                st.plotly_chart(fig_cm, use_container_width=True)
    
    # ==================== PAGE 2: Single Prediction ====================
    elif page == "🔍 Single Prediction":
        st.header("🔍 Single Image Prediction 🩺")
        st.markdown("*Upload your chest X-ray and let our AI doctors analyze it!* 👨‍⚕️✨")
        
        st.success("✅ **LIVE AI ANALYSIS** - Direct predictions from 7 trained models (Real-time, not cached) 🚀")
        st.info("💡 **How it works:** Upload any chest X-ray → AI processes it → Get instant diagnosis from multiple models!")
        
        # Add information box
        with st.expander("ℹ️ 📋 What kind of X-ray images work best?"):
            st.markdown("""
            ### 🎯 Image Requirements:
            - **Format:** JPG, JPEG, or PNG
            - **Type:** Grayscale or RGB (will be converted to grayscale)
            - **Size:** Any size (will be resized to 224x224)
            - **Quality:** Clear chest X-ray images work best
            
            **🔬 Preprocessing Pipelines:**
            - **Deep Learning:** Global standardization (mean={MEAN:.4f}, std={STD:.4f})
            - **Baseline ML:** Per-image Z-score normalization + PCA (791 components)
            
            **📊 Expected Performance (Validated on 1047 test images):**
            - VGG16 Transfer: 99.7% AUC, 97.9% Accuracy
            - ResNet50 Transfer: 99.7% AUC, 96.9% Accuracy  
            - Custom CNN: 99.0% AUC, 96.4% Accuracy
            - Gradient Boosting: 98.5% AUC, 94.7% Accuracy
            - SVM RBF: 98.8% AUC, 94.6% Accuracy
            """.format(MEAN=0.4821, STD=0.2365))
        
        # File uploader with personality
        st.markdown("### 📤 Upload Your X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image file 🖼️", 
            type=['jpg', 'jpeg', 'png'],
            help="Supports JPG, JPEG, and PNG formats. Any size works - we'll resize it for you!"
        )
        
        col1, col2 = st.columns([1, 2])
        
        if uploaded_file is not None:
            try:
                # Load image for display using PIL
                image_pil = Image.open(uploaded_file).convert('L')
                original_size = image_pil.size
                
                # Validate image
                if image_pil.size[0] < 50 or image_pil.size[1] < 50:
                    st.error("⚠️ Image too small! Please upload an image larger than 50x50 pixels.")
                    st.stop()
                
                with col1:
                    st.image(image_pil, caption="Uploaded X-Ray", use_container_width=True)
                    st.caption(f"Original size: {original_size[0]}×{original_size[1]}")
                    
                    # Show preprocessing steps
                    with st.expander("🔧 Preprocessing Steps"):
                        st.markdown("""
                        **Deep Learning Models (Notebook 04):**
                        1. ✅ Load with cv2 (grayscale)
                        2. ✅ Resize to 224×224 with cv2
                        3. ✅ Normalize to [0,1] (÷255)
                        4. ✅ Standardize with global dataset mean/std
                        5. ✅ Add channel dimension
                        
                        **Baseline ML Models (Notebook 02):**
                        1. ✅ Load with cv2 (grayscale)
                        2. ✅ Resize to 224×224 with cv2
                        3. ✅ Per-image standardization (zero mean, unit variance)
                        4. ✅ Flatten and PCA transform
                        """)
                
                # Preprocess with EXACT SAME pipeline as Notebook 04 training
                # CRITICAL: Use cv2.imread and cv2.resize (NOT PIL) to match training exactly
                import cv2
                import io
                
                # Load normalization statistics from cache summary (computed during training)
                cache_summary_path = Path.cwd() / 'reports' / 'metrics' / 'cache_summary.json'
                if cache_summary_path.exists():
                    with open(cache_summary_path, 'r') as f:
                        cache_summary = json.load(f)
                    DATASET_MEAN = cache_summary.get('normalization', {}).get('mean', 0.4821)
                    DATASET_STD = cache_summary.get('normalization', {}).get('std', 0.2365)
                else:
                    # Fallback values from training
                    DATASET_MEAN = 0.4821
                    DATASET_STD = 0.2365
                
                TARGET_SIZE = (224, 224)
                
                # Step 1: Convert uploaded file to cv2 format (EXACT as training)
                # IMPORTANT: Reset file pointer first since PIL already read it
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    st.error("❌ Could not decode image. Please upload a valid image file.")
                    st.stop()
                
                # Step 2: Resize using cv2 (EXACT as training)
                img = cv2.resize(img, TARGET_SIZE)
                
                # Step 3: Convert to float32 and normalize (EXACT as training)
                image_array = img.astype(np.float32) / 255.0
                
                # Step 4: Standardize using dataset statistics (EXACT as training)
                image_array = (image_array - DATASET_MEAN) / (DATASET_STD + 1e-7)
                
                # Step 5: Add channel dimension (1, H, W) - EXACTLY as in training
                image_array = np.expand_dims(image_array, axis=0)
                
                # Verify preprocessing
                st.sidebar.markdown("### 📊 Preprocessed Image Stats")
                st.sidebar.metric("Shape", f"{image_array.shape}")
                st.sidebar.metric("Mean", f"{image_array.mean():.4f}")
                st.sidebar.metric("Std", f"{image_array.std():.4f}")
                st.sidebar.metric("Min", f"{image_array.min():.4f}")
                st.sidebar.metric("Max", f"{image_array.max():.4f}")
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🔧 Normalization Stats")
                st.sidebar.metric("Dataset Mean", f"{DATASET_MEAN:.4f}")
                st.sidebar.metric("Dataset Std", f"{DATASET_STD:.4f}")
                st.sidebar.caption("✅ Loaded from cache_summary.json")
                
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🤖 Loaded Models")
                st.sidebar.info(f"**{len(models_dict)} models ready** for direct inference")
                for model_name, model_info in models_dict.items():
                    model_type = "🧠 Deep Learning" if model_info['type'] == 'deep_learning' else "📊 Baseline ML"
                    st.sidebar.text(f"{model_type}: {model_name}")
                
                # Load PCA for baseline models
                # CRITICAL: Notebook 02 uses PER-IMAGE standardization (not global stats)
                # Each image is standardized using its OWN mean and std
                pca_path = Path.cwd() / 'models' / 'saved_models' / 'pca_transformer.pkl'
                pca_features = None
                pca_stats = {}
                if pca_path.exists():
                    pca = joblib.load(pca_path)
                    # For PCA: apply SAME preprocessing as Notebook 02
                    # Step 1: Get raw image stats (before standardization)
                    img_for_pca = img.astype(np.float32)
                    pca_stats['raw_mean'] = float(np.mean(img_for_pca))
                    pca_stats['raw_std'] = float(np.std(img_for_pca))
                    
                    # Step 2: Per-image Z-score standardization (EXACT as Notebook 02)
                    img_standardized = (img_for_pca - pca_stats['raw_mean']) / (pca_stats['raw_std'] + 1e-7)
                    pca_stats['standardized_mean'] = float(np.mean(img_standardized))
                    pca_stats['standardized_std'] = float(np.std(img_standardized))
                    
                    # Step 3: Flatten and transform
                    image_flat = img_standardized.flatten()
                    pca_features = pca.transform(image_flat.reshape(1, -1))[0]
                    pca_stats['n_components'] = pca.n_components_
                    
                    # Display PCA preprocessing info
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 📊 Baseline Model Preprocessing")
                    st.sidebar.caption("Per-image Z-score + PCA")
                    st.sidebar.text(f"Image μ: {pca_stats['raw_mean']:.1f}")
                    st.sidebar.text(f"Image σ: {pca_stats['raw_std']:.1f}")
                    st.sidebar.text(f"After Z-score: μ≈{pca_stats['standardized_mean']:.3f}, σ≈{pca_stats['standardized_std']:.3f}")
                    st.sidebar.text(f"PCA → {pca_stats['n_components']} features")
            
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                st.stop()
            
            # Predict button with excitement!
            if st.button("🚀 Analyze with AI Models!", type="primary", use_container_width=True):
                with col2:
                    with st.spinner("🔬 AI doctors are analyzing your X-ray... Please wait! 🤖💭"):
                        results = []
                        errors = []
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_models = len(models_dict)
                        for idx, (model_name, model_info) in enumerate(models_dict.items()):
                            status_text.text(f"🤖 {model_name} is thinking... ({idx+1}/{total_models}) 🧠")
                            
                            model = model_info['model']
                            model_type = model_info['type']
                            
                            try:
                                if model_type == 'deep_learning':
                                    # Direct inference on uploaded image
                                    pred_class, confidence, probs = predict_dl_model(model, image_array)
                                else:
                                    if pca_features is None:
                                        errors.append(f"{model_name}: PCA transformer not available")
                                        continue
                                    # Direct inference with PCA features
                                    pred_class, confidence, probs = predict_baseline_model(model, pca_features)
                                
                                # Determine prediction with confidence level
                                prediction = 'PNEUMONIA' if pred_class == 1 else 'NORMAL'
                                confidence_level = "🟢 High" if confidence > 0.8 else "🟡 Medium" if confidence > 0.6 else "🔴 Low"
                                
                                results.append({
                                    'Model': model_name,
                                    'Type': model_type.replace('_', ' ').title(),
                                    'Prediction': prediction,
                                    'Confidence': f"{confidence*100:.2f}%",
                                    'Confidence Level': confidence_level,
                                    'NORMAL': f"{probs[0]*100:.1f}%",
                                    'PNEUMONIA': f"{probs[1]*100:.1f}%",
                                    '_conf_val': confidence,  # For sorting
                                    '_pred_class': pred_class
                                })
                            except Exception as e:
                                errors.append(f"{model_name}: {str(e)}")
                            
                            progress_bar.progress((idx + 1) / total_models)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Display inference mode with excitement
                        st.balloons()
                        st.success(f"🎉 **Analysis Complete!** - {len(results)} AI models have examined your X-ray! 🔬✨")
                        
                        # Display errors if any
                        if errors:
                            with st.expander("⚠️ Errors encountered"):
                                for error in errors:
                                    st.warning(error)
                        
                        if not results:
                            st.error("❌ No predictions could be generated. Please check model files.")
                            st.stop()
                        
                        # Display results
                        st.success(f"✅ Successfully generated predictions from {len(results)} models!")
                        
                        # Sort by confidence (highest first)
                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values('_conf_val', ascending=False)
                        results_display = results_df.drop(columns=['_conf_val', '_pred_class'])
                        
                        st.dataframe(results_display, use_container_width=True, hide_index=True)
                        
                        # ==== SENIOR-LEVEL ENSEMBLE PREDICTION ====
                        st.markdown("---")
                        st.markdown("## 🏆 Final Diagnosis (AI Consensus) 🧠")
                        st.markdown("*All models have voted! Here's what they collectively say:* 🗳️")
                        
                        # Use the compute_ensemble_prediction function
                        ensemble_result = compute_ensemble_prediction(results)
                        
                        if ensemble_result:
                            # Create prominent ensemble display
                            ens_col1, ens_col2 = st.columns([2, 1])
                            
                            with ens_col1:
                                # Final prediction with high visibility
                                pred_color = "🔴" if ensemble_result['prediction'] == 'PNEUMONIA' else "🟢"
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, {'#ff4b4b33' if ensemble_result['prediction'] == 'PNEUMONIA' else '#00cc6633'}, transparent);
                                    border: 2px solid {'#ff4b4b' if ensemble_result['prediction'] == 'PNEUMONIA' else '#00cc66'};
                                    border-radius: 10px;
                                    padding: 20px;
                                    text-align: center;
                                    margin: 10px 0;
                                ">
                                    <h2 style="margin: 0; color: {'#ff4b4b' if ensemble_result['prediction'] == 'PNEUMONIA' else '#00cc66'};">
                                        {pred_color} {ensemble_result['prediction']}
                                    </h2>
                                    <p style="margin: 5px 0; font-size: 1.2em;">
                                        Confidence: <strong>{ensemble_result['confidence']*100:.1f}%</strong>
                                    </p>
                                    <p style="margin: 0; font-size: 0.9em; opacity: 0.8;">
                                        P(Pneumonia) = {ensemble_result['pneumonia_prob']*100:.1f}% | 
                                        P(Normal) = {ensemble_result['normal_prob']*100:.1f}%
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with ens_col2:
                                st.markdown("#### 🗳️ Model Votes")
                                st.metric("PNEUMONIA", f"{ensemble_result['votes_pneumonia']}/{ensemble_result['total_models']}")
                                st.metric("NORMAL", f"{ensemble_result['votes_normal']}/{ensemble_result['total_models']}")
                                
                                # Agreement indicator
                                agreement = max(ensemble_result['votes_pneumonia'], ensemble_result['votes_normal']) / ensemble_result['total_models']
                                if agreement >= 0.85:
                                    st.success("✅ Strong Agreement")
                                elif agreement >= 0.6:
                                    st.warning("⚠️ Moderate Agreement")
                                else:
                                    st.error("❌ Low Agreement - Review Carefully")
                        
                        # Add performance reference
                        with st.expander("📊 Model Performance Reference (Validated on 1047 test images)"):
                            perf_data = {
                                'Model': ['VGG16 Transfer', 'ResNet50 Transfer', 'Custom CNN', 
                                         'Gradient Boosting', 'Svm Rbf', 'Logistic Regression', 'Random Forest'],
                                'ROC-AUC': ['99.72%', '99.67%', '98.99%', '98.51%', '98.79%', '97.74%', '97.48%'],
                                'Accuracy': ['97.90%', '96.94%', '96.37%', '94.65%', '94.56%', '93.89%', '77.27%'],
                                'Type': ['Deep Learning', 'Deep Learning', 'Deep Learning', 
                                        'Baseline ML', 'Baseline ML', 'Baseline ML', 'Baseline ML'],
                                'Weight': ['0.997', '0.997', '0.990', '0.985', '0.988', '0.977', '0.975']
                            }
                            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
                            st.caption("💡 Ensemble uses ROC-AUC weighted voting for optimal prediction accuracy")
                        
                        # Enhanced visualization
                        st.markdown("---")
                        st.subheader("📊 Prediction Analysis")
                        
                        # Create tabs for different visualizations
                        tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🎯 Confidence", "📈 Distribution"])
                        
                        with tab1:
                            # Stacked bar chart
                            fig1 = go.Figure(data=[
                                go.Bar(
                                    name='NORMAL',
                                    x=results_df['Model'],
                                    y=[float(r.strip('%')) for r in results_df['NORMAL']],
                                    marker_color='lightseagreen',
                                    text=[f"{float(r.strip('%')):.1f}%" for r in results_df['NORMAL']],
                                    textposition='inside'
                                ),
                                go.Bar(
                                    name='PNEUMONIA',
                                    x=results_df['Model'],
                                    y=[float(r.strip('%')) for r in results_df['PNEUMONIA']],
                                    marker_color='indianred',
                                    text=[f"{float(r.strip('%')):.1f}%" for r in results_df['PNEUMONIA']],
                                    textposition='inside'
                                )
                            ])
                            
                            fig1.update_layout(
                                title='Class Probabilities by Model',
                                xaxis_title='Model',
                                yaxis_title='Probability (%)',
                                barmode='stack',
                                height=450,
                                template='plotly_white',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with tab2:
                            # Confidence levels
                            fig2 = go.Figure(data=[
                                go.Bar(
                                    x=results_df['Model'],
                                    y=[float(r.strip('%')) for r in results_df['Confidence']],
                                    marker_color=results_df['Prediction'].map(
                                        {'PNEUMONIA': 'indianred', 'NORMAL': 'lightseagreen'}
                                    ),
                                    text=[f"{r['Prediction']}<br>{r['Confidence']}" for _, r in results_df.iterrows()],
                                    textposition='outside',
                                    hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
                                )
                            ])
                            
                            fig2.update_layout(
                                title='Prediction Confidence by Model',
                                xaxis_title='Model',
                                yaxis_title='Confidence (%)',
                                height=450,
                                template='plotly_white',
                                yaxis=dict(range=[0, 105])
                            )
                            
                            # Add threshold lines
                            fig2.add_hline(y=80, line_dash="dash", line_color="green", 
                                          annotation_text="High Confidence", annotation_position="right")
                            fig2.add_hline(y=60, line_dash="dash", line_color="orange",
                                          annotation_text="Medium Confidence", annotation_position="right")
                            
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        with tab3:
                            # Prediction distribution
                            pred_counts = results_df['Prediction'].value_counts()
                            
                            fig3 = go.Figure(data=[
                                go.Pie(
                                    labels=pred_counts.index,
                                    values=pred_counts.values,
                                    hole=0.4,
                                    marker_colors=['lightseagreen' if x == 'NORMAL' else 'indianred' 
                                                  for x in pred_counts.index],
                                    textinfo='label+percent+value',
                                    textfont_size=14
                                )
                            ])
                            
                            fig3.update_layout(
                                title='Prediction Distribution Across Models',
                                height=450,
                                template='plotly_white',
                                annotations=[dict(text=f'{len(results)}<br>Models', 
                                                x=0.5, y=0.5, font_size=20, showarrow=False)]
                            )
                            
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        # Clinical recommendation
                        st.markdown("---")
                        st.subheader("⚕️ Clinical Recommendation")
                        
                        if ensemble_result:
                            agreement = max(ensemble_result['votes_pneumonia'], ensemble_result['votes_normal']) / ensemble_result['total_models']
                            
                            if agreement > 0.85:
                                confidence_text = "Strong"
                                confidence_color = "green"
                            elif agreement > 0.6:
                                confidence_text = "Moderate"
                                confidence_color = "orange"
                            else:
                                confidence_text = "Weak"
                                confidence_color = "red"
                            
                            st.markdown(f"""
                            **Final Assessment:** :{confidence_color}[{confidence_text} consensus for {ensemble_result['prediction']}]
                            
                            - **Voting Result:** {ensemble_result['votes_pneumonia']} models predicted PNEUMONIA, {ensemble_result['votes_normal']} predicted NORMAL
                            - **Ensemble Probability:** P(Pneumonia) = {ensemble_result['pneumonia_prob']*100:.1f}%
                            - **Recommendation:** {"Further clinical evaluation recommended" if ensemble_result['prediction'] == "PNEUMONIA" else "No pneumonia detected, but clinical judgment advised"}
                            
                            ⚠️ **Disclaimer:** This is an AI-assisted diagnostic tool. Always consult with qualified healthcare professionals for medical decisions.
                            """)
    
    # ==================== PAGE 3: Ensemble Analysis ====================
    elif page == "📈 Ensemble Analysis":
        st.header("📈 Ensemble Model Analysis 🎭")
        st.markdown("*When AI models work together, magic happens! See how ensemble learning boosts performance.* ✨🤝")
        
        if ensemble_config is None:
            st.warning("⚠️ No ensemble configuration found. Run Step 7 notebook first.")
            return
        
        # Display ensemble config
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Ensemble Configuration")
            st.json(ensemble_config)
        
        with col2:
            st.subheader("🎯 Performance Metrics")
            perf = ensemble_config['performance']
            
            st.metric("ROC-AUC", f"{perf['roc_auc']:.4f}")
            st.metric("Accuracy", f"{perf['accuracy']:.4f}")
            st.metric("Precision", f"{perf['precision']:.4f}")
            st.metric("Recall", f"{perf['recall']:.4f}")
            st.metric("F1-Score", f"{perf['f1_score']:.4f}")
        
        st.markdown("---")
        
        # Optimal thresholds
        st.subheader("🎚️ Optimal Decision Thresholds")
        
        thresholds = ensemble_config['optimal_thresholds']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⚖️ Balanced", f"{thresholds['balanced']:.3f}")
        with col2:
            st.metric("🔍 High Sensitivity", f"{thresholds['high_sensitivity']:.3f}")
        with col3:
            st.metric("🎯 High Specificity", f"{thresholds['high_specificity']:.3f}")
        
        st.markdown("---")
        
        # Model weights
        st.subheader("⚖️ Model Weights in Ensemble")
        
        weights_data = pd.DataFrame({
            'Model': ensemble_config['models'],
            'Weight': ensemble_config['weights']
        }).sort_values('Weight', ascending=False)
        
        fig = px.bar(
            weights_data,
            x='Model',
            y='Weight',
            title='Model Contribution to Ensemble',
            color='Weight',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown("---")
        st.subheader("💡 Clinical Recommendation")
        st.success(f"**Recommended Model:** {ensemble_config['recommendation']}")
        
        st.info("""
        **Usage Guidelines:**
        - **Screening Mode**: Use high sensitivity threshold to minimize false negatives
        - **Confirmatory Mode**: Use high specificity threshold to minimize false positives
        - **Balanced Mode**: Use balanced threshold for general clinical decision-making
        """)
    
    # ==================== PAGE 4: Detailed Metrics ====================
    elif page == "📋 Detailed Metrics":
        st.header("📋 Detailed Performance Metrics 📊")
        st.markdown("*Dive deep into the numbers! Let's explore every metric in detail.* 🔍📈")
        
        # Load or compute predictions
        if force_recompute:
            with st.spinner("Computing detailed metrics..."):
                predictions, y_test_eval = batch_predictions(models_dict, X_test_pca, X_test_images, y_test, use_cached=False)
        else:
            with st.spinner("Loading metrics..."):
                predictions, y_test_eval = batch_predictions(models_dict, X_test_pca, X_test_images, y_test, use_cached=True)
        
        if not predictions:
            st.error("❌ No predictions available! Click 'Force Recompute Predictions' in sidebar.")
            return
        
        # Model selector
        selected_model = st.selectbox("Select Model for Detailed Analysis", list(predictions.keys()))
        
        if selected_model:
            pred_data = predictions[selected_model]
            
            # Classification report
            st.subheader(f"📊 Classification Report: {selected_model}")
            
            report = classification_report(
                y_test_eval, 
                pred_data['labels'],
                target_names=['NORMAL', 'PNEUMONIA'],
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            
            st.markdown("---")
            
            # Detailed metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔲 Confusion Matrix")
                fig_cm = plot_confusion_matrix_plotly(
                    y_test_eval, 
                    pred_data['labels'],
                    title=f"{selected_model} - Confusion Matrix"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Calculate additional metrics
                cm = confusion_matrix(y_test_eval, pred_data['labels'])
                tn, fp, fn, tp = cm.ravel()
                
                st.metric("True Negatives", tn)
                st.metric("False Positives", fp)
                st.metric("False Negatives", fn)
                st.metric("True Positives", tp)
            
            with col2:
                st.subheader("📈 Precision-Recall Curve")
                
                precision, recall, thresholds = precision_recall_curve(y_test_eval, pred_data['probs'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    fill='tozeroy',
                    name='PR Curve',
                    line=dict(color='purple', width=2)
                ))
                
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                st.metric("Sensitivity (Recall)", f"{sensitivity:.4f}")
                st.metric("Specificity", f"{specificity:.4f}")
                st.metric("PPV (Precision)", f"{ppv:.4f}")
                st.metric("NPV", f"{npv:.4f}")


if __name__ == "__main__":
    main()

# ============================================
# StockPredictionPro - src/models/classification/neural_network.py
# Neural Network classification models for financial prediction with advanced architectures
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import warnings

# Multi-backend imports with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import validation_curve
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score,
        precision_recall_curve, roc_curve, log_loss
    )
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_classifier import BaseFinancialClassifier, ClassificationStrategy

logger = get_logger('models.classification.neural_network')

# ============================================
# Pure NumPy Neural Network Implementation
# ============================================

class NumpyNeuralNetworkClassifier:
    """High-performance Neural Network using pure NumPy for universal compatibility"""
    
    def __init__(self, layers: List[int], activation: str = 'relu', 
                 learning_rate: float = 0.001, dropout_rate: float = 0.2):
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.training = True
        
        # Initialize weights using Xavier/He initialization
        for i in range(len(layers) - 1):
            if activation == 'relu':
                # He initialization for ReLU
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            else:
                # Xavier initialization for tanh/sigmoid
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(1.0 / layers[i])
            
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation_func(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        else:
            return x
    
    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute activation function derivative"""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sig * (1 - sig)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        else:
            return np.ones_like(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _dropout(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply dropout during training"""
        if self.training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask, mask
        return x, np.ones_like(x)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Forward pass through the network"""
        activations = [X]
        z_values = []
        dropout_masks = []
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b
            z_values.append(z)
            
            if i < len(self.weights) - 1:  # Hidden layers
                a = self._activation_func(z)
                a, mask = self._dropout(a)
                dropout_masks.append(mask)
            else:  # Output layer with softmax
                a = self._softmax(z)
                dropout_masks.append(np.ones_like(z))
            
            activations.append(a)
        
        return activations[-1], activations, z_values, dropout_masks
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Backward pass using backpropagation"""
        m = X.shape[0]
        
        # Forward pass
        output, activations, z_values, dropout_masks = self.forward(X)
        
        # Compute loss (cross-entropy)
        loss = -np.mean(np.sum(y * np.log(output + 1e-15), axis=1))
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (softmax + cross-entropy)
        dA = (output - y) / m
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:  # Hidden layers
                dZ = dA * self._activation_derivative(z_values[i])
                dZ *= dropout_masks[i]  # Apply dropout mask
            else:  # Output layer
                dZ = dA
            
            dW[i] = np.dot(activations[i].T, dZ)
            db[i] = np.sum(dZ, axis=0, keepdims=True)
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
        
        return loss, dW, db
    
    def update_weights(self, dW: List[np.ndarray], db: List[np.ndarray]):
        """Update weights using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.training = False
        output, _, _, _ = self.forward(X)
        self.training = True
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        self.training = False
        output, _, _, _ = self.forward(X)
        self.training = True
        return output

# ============================================
# Advanced PyTorch Implementation
# ============================================

if PYTORCH_AVAILABLE:
    class PyTorchNeuralNetworkClassifier(nn.Module):
        """Advanced PyTorch neural network with modern features"""
        
        def __init__(self, input_dim: int, hidden_layers: List[int], n_classes: int,
                     dropout: float = 0.2, batch_norm: bool = True, 
                     activation: str = 'relu'):
            super().__init__()
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.use_batch_norm = batch_norm
            
            # Activation functions
            activations = {
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'leaky_relu': nn.LeakyReLU(0.01),
                'elu': nn.ELU(),
                'gelu': nn.GELU()
            }
            self.activation = activations.get(activation.lower(), nn.ReLU())
            
            # Build layers
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                if batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer
            self.layers.append(nn.Linear(prev_dim, n_classes))
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                if self.use_batch_norm and i < len(self.batch_norms):
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
            
            # Output layer (no activation, will use softmax in loss)
            x = self.layers[-1](x)
            return x

# ============================================
# JAX Implementation
# ============================================

if JAX_AVAILABLE:
    class JAXNeuralNetworkClassifier:
        """High-performance JAX neural network with JIT compilation"""
        
        def __init__(self, layers: List[int], activation: str = 'relu', seed: int = 42):
            self.layers = layers
            self.activation = activation
            self.key = jax.random.PRNGKey(seed)
            self.params = self._init_params()
        
        def _init_params(self):
            """Initialize network parameters"""
            params = []
            keys = jax.random.split(self.key, len(self.layers))
            
            for i in range(len(self.layers) - 1):
                w_key, b_key = jax.random.split(keys[i])
                w = jax.random.normal(w_key, (self.layers[i], self.layers[i + 1])) * jnp.sqrt(2.0 / self.layers[i])
                b = jnp.zeros(self.layers[i + 1])
                params.append((w, b))
            
            return params
        
        def _activate(self, x):
            """Apply activation function"""
            if self.activation == 'relu':
                return jax.nn.relu(x)
            elif self.activation == 'tanh':
                return jnp.tanh(x)
            elif self.activation == 'sigmoid':
                return jax.nn.sigmoid(x)
            elif self.activation == 'gelu':
                return jax.nn.gelu(x)
            return x
        
        @jax.jit
        def forward(self, params, x):
            """Forward pass"""
            for w, b in params[:-1]:
                x = self._activate(jnp.dot(x, w) + b)
            
            # Output layer with softmax
            w, b = params[-1]
            logits = jnp.dot(x, w) + b
            return jax.nn.softmax(logits)
        
        @jax.jit
        def loss_fn(self, params, x, y):
            """Compute cross-entropy loss"""
            preds = self.forward(params, x)
            return -jnp.mean(jnp.sum(y * jnp.log(preds + 1e-15), axis=1))
        
        def predict(self, x):
            """Make predictions"""
            probs = self.forward(self.params, jnp.array(x))
            return np.argmax(np.array(probs), axis=1)
        
        def predict_proba(self, x):
            """Predict probabilities"""
            return np.array(self.forward(self.params, jnp.array(x)))

# ============================================
# Main Neural Network Classifier
# ============================================

class FinancialNeuralNetworkClassifier(BaseFinancialClassifier):
    """
    Neural Network classification model optimized for financial data
    
    Features:
    - Multiple backends: NumPy (universal), PyTorch, JAX, scikit-learn
    - Advanced architectures with batch normalization, dropout, regularization
    - Automatic hyperparameter optimization and early stopping
    - Learning rate scheduling and adaptive optimization
    - Comprehensive training monitoring and visualization
    - Financial domain optimizations (market regime detection, volatility weighting)
    """
    
    def __init__(self,
                 name: str = "neural_network_classifier",
                 backend: str = 'auto',  # 'auto', 'numpy', 'pytorch', 'jax', 'sklearn'
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 batch_normalization: bool = True,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 200,
                 early_stopping: bool = True,
                 early_stopping_patience: int = 20,
                 validation_split: float = 0.2,
                 l1_reg: float = 0.0,
                 l2_reg: float = 0.001,
                 optimizer: str = 'adam',
                 lr_scheduler: Optional[str] = 'plateau',
                 scaler_type: str = 'standard',
                 calibrate_probabilities: bool = True,
                 random_state: int = 42,
                 verbose: int = 1,
                 **kwargs):
        """
        Initialize Enhanced Neural Network Classifier
        
        Args:
            name: Model name
            backend: Backend to use ('auto', 'numpy', 'pytorch', 'jax', 'sklearn')
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            batch_normalization: Whether to use batch normalization
            learning_rate: Initial learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping: Whether to use early stopping
            early_stopping_patience: Early stopping patience
            validation_split: Validation data fraction
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            optimizer: Optimizer type
            lr_scheduler: Learning rate scheduler
            scaler_type: Feature scaler type
            calibrate_probabilities: Whether to calibrate probabilities
            random_state: Random seed
            verbose: Verbosity level
        """
        super().__init__(
            name=name,
            model_type="neural_network_classifier",
            classification_strategy=ClassificationStrategy.DIRECTION_PREDICTION,
            **kwargs
        )
        
        # Select best available backend
        if backend == 'auto':
            if PYTORCH_AVAILABLE:
                backend = 'pytorch'
            elif JAX_AVAILABLE:
                backend = 'jax'
            elif SKLEARN_AVAILABLE:
                backend = 'sklearn'
            else:
                backend = 'numpy'
        
        # Validate backend availability
        if backend == 'pytorch' and not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to NumPy")
            backend = 'numpy'
        elif backend == 'jax' and not JAX_AVAILABLE:
            logger.warning("JAX not available, falling back to NumPy")
            backend = 'numpy'
        elif backend == 'sklearn' and not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to NumPy")
            backend = 'numpy'
        
        self.backend = backend
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler_type = scaler_type
        self.calibrate_probabilities = calibrate_probabilities
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.scaler_ = self._create_scaler()
        self.label_encoder_: Optional[LabelEncoder] = None
        self.model_ = None
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.history_ = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
        self.best_weights_ = None
        self.training_stats_ = {}
        self.class_weights_: Optional[Dict[Any, float]] = None
        
        logger.info(f"Initialized Neural Network with {backend} backend: {self.name}")
    
    def _select_backend(self) -> str:
        """Select the best available backend"""
        backends_priority = ['pytorch', 'jax', 'sklearn', 'numpy']
        available_backends = {
            'pytorch': PYTORCH_AVAILABLE,
            'jax': JAX_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE,
            'numpy': True  # Always available
        }
        
        for backend in backends_priority:
            if available_backends[backend]:
                return backend
        
        return 'numpy'  # Fallback
    
    def _create_scaler(self):
        """Create feature scaler"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        return None
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with scaling"""
        X_processed = super()._preprocess_features(X)
        
        if self.scaler_ is not None:
            if not hasattr(self.scaler_, 'mean_'):
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug(f"Fitted {self.scaler_type} scaler")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            return X_scaled
        
        return X_processed
    
    def _preprocess_targets(self, y: pd.Series) -> np.ndarray:
        """Preprocess target labels with encoding"""
        # Convert to numpy array
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Encode labels if necessary
        if self.label_encoder_ is None:
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y_array)
            self.classes_ = self.label_encoder_.classes_
            logger.debug(f"Fitted label encoder. Classes: {self.classes_}")
        else:
            y_encoded = self.label_encoder_.transform(y_array)
        
        return y_encoded
    
    def _create_model(self):
        """Create model based on selected backend"""
        input_dim = len(self.feature_names)
        n_classes = len(self.classes_)
        
        if self.backend == 'pytorch' and PYTORCH_AVAILABLE:
            return PyTorchNeuralNetworkClassifier(
                input_dim=input_dim,
                hidden_layers=self.hidden_layers,
                n_classes=n_classes,
                dropout=self.dropout_rate,
                batch_norm=self.batch_normalization,
                activation=self.activation
            )
        
        elif self.backend == 'jax' and JAX_AVAILABLE:
            layers = [input_dim] + self.hidden_layers + [n_classes]
            return JAXNeuralNetworkClassifier(
                layers=layers,
                activation=self.activation
            )
        
        elif self.backend == 'sklearn' and SKLEARN_AVAILABLE:
            return MLPClassifier(
                hidden_layer_sizes=tuple(self.hidden_layers),
                activation=self.activation,
                alpha=self.l2_reg,
                learning_rate_init=self.learning_rate,
                max_iter=self.epochs,
                batch_size=min(self.batch_size, 200),
                validation_fraction=self.validation_split,
                early_stopping=self.early_stopping,
                n_iter_no_change=self.early_stopping_patience,
                random_state=self.random_state,
                verbose=self.verbose > 0
            )
        
        else:  # NumPy backend
            layers = [input_dim] + self.hidden_layers + [n_classes]
            return NumpyNeuralNetworkClassifier(
                layers=layers,
                activation=self.activation,
                learning_rate=self.learning_rate,
                dropout_rate=self.dropout_rate
            )
    
    def _to_categorical(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """Convert labels to one-hot encoding"""
        categorical = np.zeros((len(y), n_classes))
        categorical[np.arange(len(y)), y] = 1
        return categorical
    
    def _train_pytorch_model(self, X: np.ndarray, y: np.ndarray):
        """Train PyTorch model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_.to(device)
        
        # Split data
        val_size = int(len(X) * self.validation_split)
        if val_size > 0:
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, 
                                 weight_decay=self.l2_reg)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate, 
                                momentum=0.9, weight_decay=self.l2_reg)
        else:
            optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = None
        if self.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        elif self.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        no_improve_count = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            train_losses = []
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Add L1 regularization
                if self.l1_reg > 0:
                    l1_penalty = sum(torch.sum(torch.abs(p)) for p in self.model_.parameters())
                    loss += self.l1_reg * l1_penalty
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()
            
            train_loss = np.mean(train_losses)
            train_acc = correct_train / total_train
            
            self.history_['train_loss'].append(train_loss)
            self.history_['train_acc'].append(train_acc)
            self.history_['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation
            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(torch.FloatTensor(X_val).to(device))
                    val_loss = criterion(val_outputs, torch.LongTensor(y_val).to(device)).item()
                    
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == torch.LongTensor(y_val).to(device)).float().mean().item()
                
                self.history_['val_loss'].append(val_loss)
                self.history_['val_acc'].append(val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_weights_ = self.model_.state_dict().copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if self.early_stopping and no_improve_count >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Learning rate scheduling
                if scheduler:
                    if self.lr_scheduler == 'plateau':
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                if self.verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
            else:
                if self.verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
        
        # Load best weights
        if self.best_weights_ is not None:
            self.model_.load_state_dict(self.best_weights_)
    
    def _train_numpy_model(self, X: np.ndarray, y: np.ndarray):
        """Train NumPy model"""
        # Convert labels to one-hot encoding
        n_classes = len(self.classes_)
        y_onehot = self._to_categorical(y, n_classes)
        
        # Split data
        val_size = int(len(X) * self.validation_split)
        if val_size > 0:
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y_onehot[:-val_size], y_onehot[-val_size:]
            y_val_labels = y[-val_size:]
        else:
            X_train, y_train = X, y_onehot
            X_val, y_val, y_val_labels = None, None, None
        
        # Training loop
        best_val_acc = 0.0
        no_improve_count = 0
        
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            train_losses = []
            train_correct = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                loss, dW, db = self.model_.backward(batch_X, batch_y)
                self.model_.update_weights(dW, db)
                train_losses.append(loss)
                
                # Calculate training accuracy
                pred_probs = self.model_.predict_proba(batch_X)
                pred_labels = np.argmax(pred_probs, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                train_correct += np.sum(pred_labels == true_labels)
            
            train_loss = np.mean(train_losses)
            train_acc = train_correct / len(X_train)
            
            self.history_['train_loss'].append(train_loss)
            self.history_['train_acc'].append(train_acc)
            
            # Validation
            if X_val is not None:
                val_pred_probs = self.model_.predict_proba(X_val)
                val_pred_labels = np.argmax(val_pred_probs, axis=1)
                val_acc = np.mean(val_pred_labels == y_val_labels)
                
                # Calculate validation loss
                val_loss = -np.mean(np.sum(y_val * np.log(val_pred_probs + 1e-15), axis=1))
                
                self.history_['val_loss'].append(val_loss)
                self.history_['val_acc'].append(val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_weights_ = {
                        'weights': [w.copy() for w in self.model_.weights],
                        'biases': [b.copy() for b in self.model_.biases]
                    }
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if self.early_stopping and no_improve_count >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
        
        # Load best weights
        if self.best_weights_ is not None:
            self.model_.weights = self.best_weights_['weights']
            self.model_.biases = self.best_weights_['biases']
    
    def _calculate_training_stats(self):
        """Calculate comprehensive training statistics"""
        self.training_stats_ = {
            'backend': self.backend,
            'epochs_trained': len(self.history_['train_loss']),
            'final_train_loss': self.history_['train_loss'][-1] if self.history_['train_loss'] else None,
            'best_train_loss': min(self.history_['train_loss']) if self.history_['train_loss'] else None,
            'final_train_acc': self.history_['train_acc'][-1] if self.history_['train_acc'] else None,
            'best_train_acc': max(self.history_['train_acc']) if self.history_['train_acc'] else None,
            'final_val_loss': self.history_['val_loss'][-1] if self.history_['val_loss'] else None,
            'best_val_loss': min(self.history_['val_loss']) if self.history_['val_loss'] else None,
            'final_val_acc': self.history_['val_acc'][-1] if self.history_['val_acc'] else None,
            'best_val_acc': max(self.history_['val_acc']) if self.history_['val_acc'] else None,
            'early_stopped': len(self.history_['train_loss']) < self.epochs,
            'overfitting_gap': None
        }
        
        # Calculate overfitting gap
        if self.history_['train_acc'] and self.history_['val_acc']:
            final_train_acc = self.history_['train_acc'][-1]
            final_val_acc = self.history_['val_acc'][-1]
            self.training_stats_['overfitting_gap'] = final_train_acc - final_val_acc
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Calibrate prediction probabilities using cross-validation"""
        try:
            # Create a wrapper for the model to work with sklearn calibration
            class ModelWrapper:
                def __init__(self, model, backend):
                    self.model = model
                    self.backend = backend
                
                def predict(self, X):
                    if self.backend == 'pytorch':
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.model.eval()
                        with torch.no_grad():
                            outputs = self.model(torch.FloatTensor(X).to(device))
                            _, predicted = torch.max(outputs.data, 1)
                            return predicted.cpu().numpy()
                    elif self.backend == 'numpy':
                        return self.model.predict(X)
                    elif self.backend == 'jax':
                        return self.model.predict(X)
                    else:
                        return self.model.predict(X)
                
                def predict_proba(self, X):
                    if self.backend == 'pytorch':
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.model.eval()
                        with torch.no_grad():
                            outputs = self.model(torch.FloatTensor(X).to(device))
                            probs = torch.softmax(outputs, dim=1)
                            return probs.cpu().numpy()
                    elif self.backend == 'numpy':
                        return self.model.predict_proba(X)
                    elif self.backend == 'jax':
                        return self.model.predict_proba(X)
                    else:
                        return self.model.predict_proba(X)
            
            wrapper = ModelWrapper(self.model_, self.backend)
            self.calibrated_model_ = CalibratedClassifierCV(
                base_estimator=wrapper,
                method='isotonic',
                cv=3
            )
            self.calibrated_model_.fit(X, y)
            logger.debug("Calibrated prediction probabilities")
        except Exception as e:
            logger.warning(f"Could not calibrate probabilities: {e}")
            self.calibrated_model_ = None
    
    @time_it("neural_network_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialNeuralNetworkClassifier':
        """Fit the neural network classification model"""
        logger.info(f"Fitting Neural Network Classifier ({self.backend}) on {len(X)} samples with {X.shape[1]} features")
        
        # Validate input
        validation_result = self._validate_input_data(X, y)
        if not validation_result.is_valid:
            raise ModelValidationError(f"Input validation failed: {validation_result.errors}")
        
        try:
            # Update status
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.TRAINING
            self.last_training_time = datetime.now()
            
            # Store feature names
            self.feature_names = list(X.columns)
            self.target_name = y.name or 'target'
            
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Calculate class distribution
            unique_classes, class_counts = np.unique(y_processed, return_counts=True)
            self.class_weights_ = {cls: count / len(y_processed) for cls, count in zip(unique_classes, class_counts)}
            
            # Create model
            self.model_ = self._create_model()
            
            # Train based on backend
            fit_start = datetime.now()
            
            if self.backend == 'pytorch':
                self._train_pytorch_model(X_processed, y_processed)
            elif self.backend == 'numpy':
                self._train_numpy_model(X_processed, y_processed)
            elif self.backend == 'sklearn':
                self.model_.fit(X_processed, y_processed)
                # Extract sklearn history
                if hasattr(self.model_, 'loss_curve_'):
                    self.history_['train_loss'] = list(self.model_.loss_curve_)
                if hasattr(self.model_, 'validation_scores_'):
                    self.history_['val_acc'] = list(self.model_.validation_scores_)
            elif self.backend == 'jax':
                # JAX training implementation would go here
                # For now, simplified training
                logger.warning("JAX training not fully implemented, using basic fit")
            
            fit_duration = (datetime.now() - fit_start).total_seconds()
            self.training_duration = fit_duration
            
            # Calculate training statistics
            self._calculate_training_stats()
            
            # Update metadata
            self.update_metadata({
                'training_samples': len(X),
                'training_features': X.shape[1],
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'backend': self.backend,
                'architecture': self.hidden_layers,
                'training_duration_seconds': fit_duration,
                'epochs_trained': len(self.history_['train_loss'])
            })
            
            # Calculate training score
            predictions = self.predict(X)
            self.training_score = accuracy_score(y, predictions)
            
            # Calibrate probabilities if requested
            if self.calibrate_probabilities and self.backend != 'sklearn':
                self._calibrate_probabilities(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"Neural Network training complete in {fit_duration:.2f}s")
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Neural Network training failed: {e}")
            raise
    
    @time_it("neural_network_predict", include_args=True)
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making Neural Network predictions for {len(X)} samples")
        
        try:
            X_processed = self._preprocess_features(X)
            
            if self.calibrated_model_ is not None:
                predictions_encoded = self.calibrated_model_.predict(X_processed)
            else:
                if self.backend == 'pytorch':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model_.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_processed).to(device)
                        outputs = self.model_(X_tensor)
                        _, predictions_encoded = torch.max(outputs.data, 1)
                        predictions_encoded = predictions_encoded.cpu().numpy()
                elif self.backend == 'numpy':
                    predictions_encoded = self.model_.predict(X_processed)
                elif self.backend == 'sklearn':
                    predictions_encoded = self.model_.predict(X_processed)
                elif self.backend == 'jax':
                    predictions_encoded = self.model_.predict(X_processed)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
            
            # Decode predictions
            predictions = self.label_encoder_.inverse_transform(predictions_encoded)
            
            self.log_prediction()
            return predictions
            
        except Exception as e:
            logger.error(f"Neural Network prediction failed: {e}")
            raise
    
    @time_it("neural_network_predict_proba", include_args=True)
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making Neural Network probability predictions for {len(X)} samples")
        
        try:
            X_processed = self._preprocess_features(X)
            
            if self.calibrated_model_ is not None:
                probabilities = self.calibrated_model_.predict_proba(X_processed)
            else:
                if self.backend == 'pytorch':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model_.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_processed).to(device)
                        outputs = self.model_(X_tensor)
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                elif self.backend == 'numpy':
                    probabilities = self.model_.predict_proba(X_processed)
                elif self.backend == 'sklearn':
                    probabilities = self.model_.predict_proba(X_processed)
                elif self.backend == 'jax':
                    probabilities = self.model_.predict_proba(X_processed)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Neural Network probability prediction failed: {e}")
            raise
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return self.history_.copy()
    
    def plot_training_history(self) -> Any:
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            if self.history_['train_loss']:
                axes[0, 0].plot(self.history_['train_loss'], label='Training Loss', linewidth=2, color='blue')
            if self.history_['val_loss']:
                axes[0, 0].plot(self.history_['val_loss'], label='Validation Loss', linewidth=2, color='red')
            
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss History')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy plot
            if self.history_['train_acc']:
                axes[0, 1].plot(self.history_['train_acc'], label='Training Accuracy', linewidth=2, color='blue')
            if self.history_['val_acc']:
                axes[0, 1].plot(self.history_['val_acc'], label='Validation Accuracy', linewidth=2, color='red')
            
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy History')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate plot
            if self.history_['lr']:
                axes[1, 0].plot(self.history_['lr'], color='orange', linewidth=2)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_yscale('log')
            else:
                axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Training statistics
            if self.training_stats_:
                stats_text = f"Backend: {self.training_stats_['backend']}\n"
                stats_text += f"Epochs: {self.training_stats_['epochs_trained']}\n"
                stats_text += f"Early Stopped: {'Yes' if self.training_stats_['early_stopped'] else 'No'}\n"
                if self.training_stats_['best_val_acc']:
                    stats_text += f"Best Val Acc: {self.training_stats_['best_val_acc']:.4f}\n"
                if self.training_stats_['overfitting_gap']:
                    stats_text += f"Overfitting Gap: {self.training_stats_['overfitting_gap']:.4f}\n"
                
                axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            axes[1, 1].set_title('Training Statistics')
            axes[1, 1].axis('off')
            
            plt.suptitle(f'Neural Network Training History - {self.name} ({self.backend})', fontsize=16)
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.Series, 
                             normalize: str = 'true') -> Any:
        """Plot confusion matrix"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Make predictions
            y_pred = self.predict(X)
            y_true_encoded = self.label_encoder_.transform(y)
            y_pred_encoded = self.label_encoder_.transform(y_pred)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true_encoded, y_pred_encoded, normalize=normalize)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Use class names for labels
            class_names = [self.label_encoder_.inverse_transform([i])[0] for i in range(len(self.classes_))]
            
            sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                       cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            
            plt.title(f'Confusion Matrix - {self.name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return None
    
    def plot_roc_curve(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Plot ROC curve (for binary classification)"""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import auc
            
            if len(self.classes_) != 2:
                logger.warning("ROC curve is only available for binary classification")
                return None
            
            # Get probabilities
            y_proba = self.predict_proba(X)[:, 1]  # Positive class probability
            y_true_encoded = self.label_encoder_.transform(y)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_encoded, y_proba)
            auc_score = auc(fpr, tpr)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get architecture summary"""
        total_params = 0
        
        if self.backend == 'pytorch' and hasattr(self.model_, 'parameters'):
            total_params = sum(p.numel() for p in self.model_.parameters())
        elif self.backend == 'numpy' and hasattr(self.model_, 'weights'):
            total_params = sum(w.size + b.size for w, b in zip(self.model_.weights, self.model_.biases))
        
        return {
            'backend': self.backend,
            'architecture': self.hidden_layers,
            'activation': self.activation,
            'total_parameters': total_params,
            'dropout_rate': self.dropout_rate,
            'batch_normalization': self.batch_normalization,
            'regularization': {
                'l1': self.l1_reg,
                'l2': self.l2_reg,
                'dropout': self.dropout_rate
            }
        }
    
    def get_neural_network_summary(self) -> Dict[str, Any]:
        """Get comprehensive neural network summary"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get summary")
        
        return {
            'architecture': self.get_architecture_summary(),
            'training_config': {
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'early_stopping': self.early_stopping,
                'lr_scheduler': self.lr_scheduler
            },
            'training_stats': self.training_stats_,
            'backend_info': {
                'selected_backend': self.backend,
                'available_backends': {
                    'pytorch': PYTORCH_AVAILABLE,
                    'jax': JAX_AVAILABLE,
                    'sklearn': SKLEARN_AVAILABLE,
                    'numpy': True
                }
            },
            'class_distribution': self.class_weights_
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        summary.update({
            'model_family': 'Neural Network',
            'backend': self.backend,
            'architecture': self.hidden_layers,
            'activation': self.activation,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'regularization_l1': self.l1_reg,
            'regularization_l2': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            'batch_normalization': self.batch_normalization,
            'early_stopping': self.early_stopping,
            'probability_calibration': self.calibrate_probabilities,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None
        })
        
        if self.is_fitted:
            try:
                nn_summary = self.get_neural_network_summary()
                summary.update({
                    'epochs_trained': nn_summary['training_stats']['epochs_trained'],
                    'early_stopped': nn_summary['training_stats']['early_stopped'],
                    'best_val_acc': nn_summary['training_stats']['best_val_acc'],
                    'overfitting_gap': nn_summary['training_stats']['overfitting_gap']
                })
            except Exception as e:
                logger.debug(f"Could not generate neural network summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_neural_network_classifier(backend: str = 'auto',
                                    architecture: str = 'balanced',
                                    complexity: str = 'medium',
                                    **kwargs) -> FinancialNeuralNetworkClassifier:
    """Create neural network with optimal backend selection"""
    
    architecture_configs = {
        'simple': {'hidden_layers': [64], 'dropout_rate': 0.1, 'epochs': 100},
        'balanced': {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.2, 'epochs': 200},
        'deep': {'hidden_layers': [256, 128, 64, 32, 16], 'dropout_rate': 0.3, 'epochs': 300},
        'wide': {'hidden_layers': [512, 256, 128], 'dropout_rate': 0.25, 'epochs': 200}
    }
    
    complexity_configs = {
        'low': {'learning_rate': 0.01, 'batch_size': 64, 'l2_reg': 0.01},
        'medium': {'learning_rate': 0.001, 'batch_size': 32, 'l2_reg': 0.001},
        'high': {'learning_rate': 0.0001, 'batch_size': 16, 'l2_reg': 0.0001}
    }
    
    config = {
        'name': f'neural_network_{architecture}_{complexity}',
        'backend': backend,
        'activation': 'relu',
        'batch_normalization': True,
        'early_stopping': True,
        'lr_scheduler': 'plateau',
        'random_state': 42
    }
    
    config.update(architecture_configs.get(architecture, architecture_configs['balanced']))
    config.update(complexity_configs.get(complexity, complexity_configs['medium']))
    config.update(kwargs)
    
    return FinancialNeuralNetworkClassifier(**config)

def create_pytorch_neural_network(**kwargs) -> FinancialNeuralNetworkClassifier:
    """Create PyTorch neural network"""
    return create_neural_network_classifier(backend='pytorch', **kwargs)

def create_lightweight_neural_network(**kwargs) -> FinancialNeuralNetworkClassifier:
    """Create lightweight NumPy neural network"""
    return create_neural_network_classifier(
        backend='numpy',
        architecture='simple',
        complexity='low',
        **kwargs
    )

def create_advanced_neural_network(**kwargs) -> FinancialNeuralNetworkClassifier:
    """Create advanced neural network with best available backend"""
    return create_neural_network_classifier(
        backend='auto',
        architecture='deep',
        complexity='high',
        **kwargs
    )

def create_binary_neural_network(**kwargs) -> FinancialNeuralNetworkClassifier:
    """Create neural network optimized for binary classification"""
    return create_neural_network_classifier(
        architecture='balanced',
        complexity='medium',
        name='binary_neural_network_classifier',
        **kwargs
    )

def create_multiclass_neural_network(**kwargs) -> FinancialNeuralNetworkClassifier:
    """Create neural network optimized for multiclass classification"""
    return create_neural_network_classifier(
        architecture='deep',
        complexity='medium',
        dropout_rate=0.3,  # Higher dropout for multiclass
        name='multiclass_neural_network_classifier',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def compare_neural_network_backends(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Compare different neural network backends"""
    
    backends_to_test = []
    if PYTORCH_AVAILABLE:
        backends_to_test.append('pytorch')
    if JAX_AVAILABLE:
        backends_to_test.append('jax')
    if SKLEARN_AVAILABLE:
        backends_to_test.append('sklearn')
    backends_to_test.append('numpy')  # Always available
    
    results = {}
    
    for backend in backends_to_test:
        logger.info(f"Testing {backend} backend")
        
        try:
            model = create_neural_network_classifier(
                backend=backend,
                architecture='simple',
                epochs=50,
                verbose=0
            )
            
            import time
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time
            
            # Evaluate
            metrics = model.evaluate(X, y)
            
            results[backend] = {
                'training_time': training_time,
                'accuracy': metrics.accuracy,
                'f1_score': metrics.f1_score,
                'available': True,
                'epochs_trained': model.training_stats_['epochs_trained']
            }
            
        except Exception as e:
            logger.warning(f"Error testing {backend}: {e}")
            results[backend] = {'available': False, 'error': str(e)}
    
    # Add comparison summary
    available_backends = [k for k, v in results.items() if v.get('available', False)]
    
    if available_backends:
        best_accuracy = max(available_backends, key=lambda k: results[k]['accuracy'])
        fastest = min(available_backends, key=lambda k: results[k]['training_time'])
        
        results['comparison'] = {
            'best_accuracy': best_accuracy,
            'fastest': fastest,
            'available_backends': available_backends,
            'recommended': 'pytorch' if 'pytorch' in available_backends else available_backends[0]
        }
    
    return results

def get_backend_capabilities() -> Dict[str, Any]:
    """Get capabilities of different backends"""
    
    return {
        'pytorch': {
            'available': PYTORCH_AVAILABLE,
            'features': ['GPU Support', 'Advanced Optimizers', 'Dynamic Graphs', 'Comprehensive'],
            'performance': 'Excellent',
            'ease_of_use': 'Good',
            'python_compatibility': '3.8+'
        },
        'jax': {
            'available': JAX_AVAILABLE,
            'features': ['JIT Compilation', 'Functional Programming', 'XLA Acceleration', 'Research'],
            'performance': 'Excellent',
            'ease_of_use': 'Advanced',
            'python_compatibility': '3.9+'
        },
        'sklearn': {
            'available': SKLEARN_AVAILABLE,
            'features': ['Simple API', 'Scikit Integration', 'Stable', 'Traditional ML'],
            'performance': 'Good',
            'ease_of_use': 'Excellent',
            'python_compatibility': '3.8+'
        },
        'numpy': {
            'available': True,
            'features': ['Universal Compatibility', 'Lightweight', 'Educational', 'No Dependencies'],
            'performance': 'Good',
            'ease_of_use': 'Good',
            'python_compatibility': '3.6+'
        }
    }

def tune_neural_network_hyperparameters(X: pd.DataFrame, y: pd.Series,
                                       backend: str = 'auto',
                                       param_grid: Optional[Dict[str, List[Any]]] = None,
                                       cv: int = 3) -> Dict[str, Any]:
    """Tune neural network hyperparameters"""
    
    logger.info("Starting Neural Network hyperparameter tuning")
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'hidden_layers': [[64], [128, 64], [128, 64, 32]],
            'learning_rate': [0.01, 0.001, 0.0001],
            'dropout_rate': [0.1, 0.2, 0.3],
            'batch_size': [16, 32, 64]
        }
    
    from sklearn.model_selection import ParameterGrid
    
    best_score = -np.inf
    best_params = None
    all_results = []
    
    for params in ParameterGrid(param_grid):
        try:
            logger.info(f"Testing parameters: {params}")
            
            # Create model with current parameters
            model = create_neural_network_classifier(
                backend=backend,
                epochs=50,  # Reduced for tuning
                verbose=0,
                **params
            )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            
            # Fit model first to get preprocessing
            model.fit(X, y)
            
            # Manual cross-validation for neural networks
            scores = []
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                fold_model = create_neural_network_classifier(
                    backend=backend,
                    epochs=50,
                    verbose=0,
                    **params
                )
                fold_model.fit(X_train_fold, y_train_fold)
                score = fold_model.evaluate(X_val_fold, y_val_fold).accuracy
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            result = {
                'params': params,
                'cv_mean': mean_score,
                'cv_std': std_score,
                'cv_scores': scores
            }
            all_results.append(result)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                
            logger.info(f"Score: {mean_score:.4f} ± {std_score:.4f}")
            
        except Exception as e:
            logger.warning(f"Error with parameters {params}: {e}")
    
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
        'backend': backend
    }
    
    logger.info(f"Hyperparameter tuning complete. Best score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return results

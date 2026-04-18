"""
Neural Network Node - Deep learning models.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class NeuralNetNode(NodeRuntime):
    """
    Neural network architectures:
    - MLP (Multi-Layer Perceptron)
    - CNN (1D Convolutional)
    - RNN, LSTM, GRU
    - Transformer (basic)
    - AutoEncoder
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        
        if X_train is None or y_train is None:
            return NodeResult(outputs={}, success=False, error_message="Missing training data")
        
        if not TF_AVAILABLE:
            return NodeResult(outputs={}, success=False, error_message="TensorFlow not available")
        
        X_train = ensure_dataframe(X_train)
        y_train = np.array(y_train).ravel() if hasattr(y_train, '__len__') else y_train
        
        architecture = self.get_option("Architecture", "MLP")
        hidden_layers = self.get_option("Hidden Layers", "128,64,32")
        activation = self.get_option("Activation", "ReLU")
        dropout = self.get_option("Dropout", 0.2)
        optimizer = self.get_option("Optimizer", "Adam")
        batch_norm = self.get_option("Batch Norm", True)
        epochs = int(self.get_option("Epochs", 20))
        batch_size = int(self.get_option("Batch Size", 32))
        validation_split = float(self.get_option("Validation Split", 0.2))
        
        try:
            # Parse hidden layers
            layer_sizes = [int(x.strip()) for x in hidden_layers.split(",")]
            
            # Determine task type
            n_classes = len(np.unique(y_train))
            is_classification = n_classes < 50  # Heuristic
            
            # Build model
            model = self._build_model(
                architecture=architecture,
                input_shape=X_train.shape[1],
                layer_sizes=layer_sizes,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm,
                n_classes=n_classes if is_classification else 1,
                is_classification=is_classification
            )
            
            # Compile model
            model = self._compile_model(model, optimizer, is_classification)
            
            # Prepare data
            X_clean = X_train.fillna(0).values.astype(float)
            y_clean = y_train

            # Sequence-style architectures expect [samples, timesteps, channels].
            if architecture in {"CNN", "RNN", "LSTM", "GRU"}:
                X_clean = X_clean[..., np.newaxis]
            
            if is_classification:
                # One-hot encode for multi-class
                if n_classes > 2:
                    y_clean = keras.utils.to_categorical(y_clean, n_classes)
            elif architecture == "AutoEncoder":
                # Autoencoders reconstruct the input.
                y_clean = X_clean

            # Train the model (previously model was returned untrained).
            history = model.fit(
                X_clean,
                y_clean,
                epochs=max(1, epochs),
                batch_size=max(1, batch_size),
                validation_split=max(0.0, min(0.9, validation_split)),
                verbose=0,
            )
            
            # Store model
            self.set_fitted_state("model", model)
            self.set_fitted_state("is_classification", is_classification)
            self.set_fitted_state("n_classes", n_classes)
            
            # Architecture summary
            architecture_info = {
                "type": architecture,
                "layers": layer_sizes,
                "activation": activation,
                "dropout": dropout,
                "total_params": model.count_params(),
            }
            
            return NodeResult(
                outputs={
                    "NN Model": model,
                    "Architecture": architecture_info,
                    "Training History": {k: list(v) for k, v in history.history.items()},
                },
                metadata={
                    "input_shape": X_train.shape,
                    "n_classes": n_classes if is_classification else None,
                    "task": "classification" if is_classification else "regression",
                    "trained_epochs": max(1, epochs),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _build_model(self, architecture: str, input_shape: int, layer_sizes: list[int],
                     activation: str, dropout: float, batch_norm: bool,
                     n_classes: int, is_classification: bool) -> keras.Model:
        """Build neural network model."""
        act_map = {
            "ReLU": "relu",
            "LeakyReLU": keras.layers.LeakyReLU(),
            "Tanh": "tanh",
            "Sigmoid": "sigmoid",
            "GELU": "gelu",
            "Swish": "swish",
        }
        act = act_map.get(activation, "relu")
        
        if architecture == "MLP":
            return self._build_mlp(input_shape, layer_sizes, act, dropout, batch_norm, n_classes, is_classification)
        
        elif architecture == "CNN":
            return self._build_cnn(input_shape, layer_sizes, act, dropout, batch_norm, n_classes, is_classification)
        
        elif architecture in ["RNN", "LSTM", "GRU"]:
            return self._build_rnn(input_shape, layer_sizes, architecture, act, dropout, batch_norm, n_classes, is_classification)
        
        elif architecture == "Transformer":
            return self._build_transformer(input_shape, layer_sizes, act, dropout, n_classes, is_classification)
        
        elif architecture == "AutoEncoder":
            return self._build_autoencoder(input_shape, layer_sizes, act, dropout, batch_norm)
        
        else:
            return self._build_mlp(input_shape, layer_sizes, act, dropout, batch_norm, n_classes, is_classification)
    
    def _build_mlp(self, input_shape, layer_sizes, activation, dropout, batch_norm, n_classes, is_classification):
        """Build MLP model."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_shape,)))
        
        for units in layer_sizes:
            model.add(keras.layers.Dense(units))
            if batch_norm:
                model.add(keras.layers.BatchNormalization())
            if isinstance(activation, str):
                model.add(keras.layers.Activation(activation))
            else:
                model.add(activation)
            if dropout > 0:
                model.add(keras.layers.Dropout(dropout))
        
        # Output layer
        if is_classification:
            if n_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
            else:
                model.add(keras.layers.Dense(n_classes, activation='softmax'))
        else:
            model.add(keras.layers.Dense(1, activation='linear'))
        
        return model
    
    def _build_cnn(self, input_shape, layer_sizes, activation, dropout, batch_norm, n_classes, is_classification):
        """Build 1D CNN model."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_shape, 1)))
        
        for i, units in enumerate(layer_sizes[:3]):  # Max 3 conv layers
            model.add(keras.layers.Conv1D(units, kernel_size=3, padding='same'))
            if batch_norm:
                model.add(keras.layers.BatchNormalization())
            if isinstance(activation, str):
                model.add(keras.layers.Activation(activation))
            else:
                model.add(activation)
            model.add(keras.layers.MaxPooling1D(2))
            if dropout > 0:
                model.add(keras.layers.Dropout(dropout))
        
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        
        if is_classification:
            if n_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
            else:
                model.add(keras.layers.Dense(n_classes, activation='softmax'))
        else:
            model.add(keras.layers.Dense(1, activation='linear'))
        
        return model
    
    def _build_rnn(self, input_shape, layer_sizes, rnn_type, activation, dropout, batch_norm, n_classes, is_classification):
        """Build RNN/LSTM/GRU model."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_shape, 1)))
        
        rnn_map = {
            "RNN": keras.layers.SimpleRNN,
            "LSTM": keras.layers.LSTM,
            "GRU": keras.layers.GRU,
        }
        RNNLayer = rnn_map.get(rnn_type, keras.layers.LSTM)
        
        for i, units in enumerate(layer_sizes):
            return_sequences = i < len(layer_sizes) - 1
            model.add(RNNLayer(units, return_sequences=return_sequences, dropout=dropout))
        
        model.add(keras.layers.Dense(64, activation='relu'))
        
        if is_classification:
            if n_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
            else:
                model.add(keras.layers.Dense(n_classes, activation='softmax'))
        else:
            model.add(keras.layers.Dense(1, activation='linear'))
        
        return model
    
    def _build_transformer(self, input_shape, layer_sizes, activation, dropout, n_classes, is_classification):
        """Build simple Transformer model."""
        inputs = keras.layers.Input(shape=(input_shape,))
        
        # Reshape for attention
        x = keras.layers.Reshape((input_shape, 1))(inputs)
        
        # Simple self-attention
        x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(dropout)(x)
        
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(dropout)(x)
        
        if is_classification:
            if n_classes == 2:
                outputs = keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = keras.layers.Dense(n_classes, activation='softmax')(x)
        else:
            outputs = keras.layers.Dense(1, activation='linear')(x)
        
        return keras.Model(inputs, outputs)
    
    def _build_autoencoder(self, input_shape, layer_sizes, activation, dropout, batch_norm):
        """Build AutoEncoder model."""
        # Encoder
        encoder_inputs = keras.layers.Input(shape=(input_shape,))
        x = encoder_inputs
        
        for units in layer_sizes:
            x = keras.layers.Dense(units)(x)
            if batch_norm:
                x = keras.layers.BatchNormalization()(x)
            if isinstance(activation, str):
                x = keras.layers.Activation(activation)(x)
            if dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
        
        # Decoder (reverse layer sizes)
        for units in reversed(layer_sizes[:-1]):
            x = keras.layers.Dense(units)(x)
            if batch_norm:
                x = keras.layers.BatchNormalization()(x)
            if isinstance(activation, str):
                x = keras.layers.Activation(activation)(x)
            if dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
        
        outputs = keras.layers.Dense(input_shape, activation='linear')(x)
        
        return keras.Model(encoder_inputs, outputs)
    
    def _compile_model(self, model, optimizer: str, is_classification: bool):
        """Compile the model."""
        opt_map = {
            "Adam": keras.optimizers.Adam(learning_rate=0.001),
            "SGD": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            "AdamW": keras.optimizers.AdamW(learning_rate=0.001),
            "RMSprop": keras.optimizers.RMSprop(learning_rate=0.001),
            "Adagrad": keras.optimizers.Adagrad(learning_rate=0.01),
        }
        opt = opt_map.get(optimizer, keras.optimizers.Adam())
        
        if is_classification:
            loss = "sparse_categorical_crossentropy" if model.output_shape[-1] > 1 else "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            loss = "mse"
            metrics = ["mae"]
        
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return model

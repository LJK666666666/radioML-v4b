"""
DAE (Denoising Autoencoder) Model for RadioML
Based on: AMR-Benchmark/RML201610a/DAE/rmlmodels/DAE.py

Architecture:
- Input: (128, 2) - time steps x features (I/Q)
- 2x LSTM layers (32 units each)
- Classifier branch: Dense layers -> softmax
- Decoder branch: TimeDistributed Dense -> reconstructed signal

Reference: AMR-Benchmark implementation
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization,
    TimeDistributed, Reshape
)


def build_dae_model(input_shape=(2, 128), num_classes=11, use_gpu_lstm=True):
    """
    Build DAE model for automatic modulation classification.

    Args:
        input_shape: Input shape (channels, time_steps) = (2, 128) for I/Q data
        num_classes: Number of modulation classes
        use_gpu_lstm: Whether to use GPU-optimized LSTM (CuDNNLSTM-like behavior)
                     If False, uses standard LSTM

    Returns:
        Keras Model with dual outputs: (classification, reconstruction)

    Note:
        The original model uses CuDNNLSTM which is deprecated. We use standard LSTM
        with activation='tanh', recurrent_activation='sigmoid' to match CuDNNLSTM behavior.
    """
    # Input shape needs to be transposed for LSTM
    # PyTorch/Original: (2, 128) -> need to transpose to (128, 2) for LSTM
    # LSTM expects (time_steps, features)
    if input_shape == (2, 128):
        # Need to handle transpose
        inputs = Input(shape=input_shape, name='input')
        # Transpose from (2, 128) to (128, 2)
        x = tf.keras.layers.Permute((2, 1))(inputs)  # (batch, 2, 128) -> (batch, 128, 2)
    else:
        # Already in correct format (128, 2)
        inputs = Input(shape=input_shape, name='input')
        x = inputs

    # Dropout rate
    dr = 0.0  # Set to 0 as in original

    # LSTM Unit 1: 32 units, return sequences and states
    # CuDNNLSTM equivalent: use activation='tanh', recurrent_activation='sigmoid'
    lstm_kwargs = {
        'units': 32,
        'return_state': True,
        'return_sequences': True
    }

    if use_gpu_lstm:
        # Standard LSTM with settings that match CuDNNLSTM behavior
        lstm_kwargs.update({
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'use_bias': True,
            'recurrent_dropout': 0.0
        })

    x, s, c = LSTM(**lstm_kwargs)(x)
    x = Dropout(dr)(x)

    # LSTM Unit 2: 32 units, return sequences and states
    x, s1, c1 = LSTM(**lstm_kwargs)(x)

    # Classifier branch (uses final state s1)
    xc = Dense(32, activation='relu')(s1)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(16, activation='relu')(xc)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(num_classes, activation='softmax', name='xc')(xc)

    # Decoder branch (uses full sequence x)
    xd = TimeDistributed(Dense(2), name='xd')(x)

    # Create model with dual outputs
    model = Model(inputs=inputs, outputs=[xc, xd], name='DAE')

    # Compile model
    # Loss weights: prioritize classification over reconstruction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'xc': 'categorical_crossentropy',
            'xd': 'mse'  # Mean squared error for reconstruction
        },
        loss_weights={'xc': 1.0, 'xd': 0.5},  # Classification more important
        metrics={'xc': 'accuracy'}
    )

    return model


def build_dae_model_classifier_only(input_shape=(2, 128), num_classes=11, use_gpu_lstm=True):
    """
    Build DAE model for classification only (no reconstruction output).

    This version only outputs the classification predictions, which is more
    convenient for standard classification workflows.

    Args:
        input_shape: Input shape (channels, time_steps) = (2, 128) for I/Q data
        num_classes: Number of modulation classes
        use_gpu_lstm: Whether to use GPU-optimized LSTM

    Returns:
        Keras Model with single classification output
    """
    # Input shape needs to be transposed for LSTM
    # PyTorch/Original: (2, 128) -> need to transpose to (128, 2) for LSTM
    # LSTM expects (time_steps, features)
    if input_shape == (2, 128):
        # Need to handle transpose
        inputs = Input(shape=input_shape, name='input')
        # Transpose from (2, 128) to (128, 2)
        x = tf.keras.layers.Permute((2, 1))(inputs)  # (batch, 2, 128) -> (batch, 128, 2)
    else:
        # Already in correct format (128, 2)
        inputs = Input(shape=input_shape, name='input')
        x = inputs

    # Dropout rate
    dr = 0.0  # Set to 0 as in original

    # LSTM Unit 1: 32 units, return sequences and states
    # CuDNNLSTM equivalent: use activation='tanh', recurrent_activation='sigmoid'
    lstm_kwargs = {
        'units': 32,
        'return_state': True,
        'return_sequences': True
    }

    if use_gpu_lstm:
        # Standard LSTM with settings that match CuDNNLSTM behavior
        lstm_kwargs.update({
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'use_bias': True,
            'recurrent_dropout': 0.0
        })

    x, s, c = LSTM(**lstm_kwargs)(x)
    x = Dropout(dr)(x)

    # LSTM Unit 2: 32 units, return sequences and states
    x, s1, c1 = LSTM(**lstm_kwargs)(x)

    # Classifier branch (uses final state s1)
    xc = Dense(32, activation='relu')(s1)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(16, activation='relu')(xc)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    outputs = Dense(num_classes, activation='softmax', name='classification')(xc)

    # Create model with single classification output
    model = Model(inputs=inputs, outputs=outputs, name='DAE_Classifier')

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    # Test model building
    print("Testing DAE model construction...")

    # Test with input shape (2, 128)
    print("\n1. Testing with input_shape=(2, 128)...")
    model = build_dae_model(input_shape=(2, 128), num_classes=11)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    # Test with sample input
    import numpy as np
    x = np.random.randn(4, 2, 128).astype(np.float32)
    print(f"\nInput shape: {x.shape}")
    outputs = model.predict(x, verbose=0)
    print(f"Classification output shape: {outputs[0].shape}")
    print(f"Reconstruction output shape: {outputs[1].shape}")

    # Test classifier-only version
    print("\n2. Testing classifier-only version...")
    classifier_model = build_dae_model_classifier_only(input_shape=(2, 128), num_classes=11)
    classifier_model.summary()
    y = classifier_model.predict(x, verbose=0)
    print(f"\nClassification output shape: {y.shape}")

    print("\n✓ All tests passed!")

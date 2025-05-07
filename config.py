# config.py

FTCP_CONFIG = {
    # === Encoder/Decoder Architecture ===
    "latent_dim": 256,                  # Dimensionality of latent space
    "filters": [32, 64, 128],           # Number of filters in each Conv1D layer
    "filter_sizes": [5, 3, 3],          # Kernel sizes for each Conv1D/Conv2DTranspose
    "strides": [2, 2, 1],               # Strides for each convolutional layer

    # === Activation & Normalization ===
    "leakyrelu_alpha": 0.2,             # Negative slope coefficient for LeakyReLU
    "activation_dense": "relu",         # Activation for dense layers (regression + decoder)
    "activation_latent": "sigmoid",     # Activation before latent representation
    "activation_conv": "relu",          # Activation in decoder Conv2DTranspose layers
    "activation_final_decoder": "sigmoid",  # Activation after final decoder layer
    "regression_output_activation": "sigmoid",  # Activation for regression output layer
    "use_batchnorm": True,              # Whether to apply BatchNormalization
    "use_dropout": False,               # Whether to use dropout in regression branch
    "dropout_rate": 0.3,                # Dropout rate if enabled

    # === Latent Bottleneck ===
    "dense_latent_dim": 1024,           # Size of dense layer before latent mean/log_var

    # === Regression MLP (for property prediction) ===
    "regression_hidden": [128, 32],     # List of hidden units in regression head

    # === Loss Coefficients ===
    "coeff_KL": 2,                      # KL divergence loss weight
    "coeff_prop": 10,                   # Property prediction loss weight
    "coeff_prop_semi": 5,               # Semi-supervised property loss weight

    # === Optimizer and Training ===
    "optimizer": "RMSprop",             # Optimizer class name (e.g., 'Adam', 'SGD')
    "learning_rate": 5e-4,              # Learning rate
    "batch_size": 64,                   # Training batch size
    "epochs": 200                       # Number of training epochs
}

# config.py

FTCP_CONFIG = {
    # --- Architecture ---
    "latent_dim": 256,                  # Latent space dimension
    "filters": [32, 64, 128],           # Conv1D/Conv2DTranspose filters
    "filter_sizes": [5, 3, 3],          # Kernel sizes
    "strides": [2, 2, 1],               # Strides for conv layers
    "optimizer": "RMSprop", 

    # --- Activations ---
    "activation_dense": "relu",         # Dense layers
    "activation_latent": "sigmoid",     # Latent bottleneck layer
    "activation_conv": "relu",          # Decoder conv layers
    "activation_final_decoder": "sigmoid",  # Last decoder activation
    "regression_output_activation": "sigmoid",  # Regression output layer

    # --- Regularization ---
    "use_batchnorm": True,              # Apply BatchNormalization
    "leakyrelu_alpha": 0.2,             # LeakyReLU slope (encoder)
    "use_dropout": False,               # Enable dropout (optional)
    "dropout_rate": 0.3,                # Dropout rate

    # --- Regression ---
    "regression_hidden": [128, 32],     # Hidden layers for property prediction

    # --- Loss weights ---
    "coeff_KL": 2,                      # KL divergence weight
    "coeff_prop": 10,                   # Property loss weight
    "coeff_prop_semi": 5,               # Semi-supervised property loss weight

    # --- Training ---
    "learning_rate": 5e-4,
    "batch_size": 64,
    "epochs": 200,

    # --- Sampling / Inverse Design ---
    "Lp_scale": 0.6,                    # Latent perturbation scale
    "Nperturb": 3                       # Number of perturbations per sample
}

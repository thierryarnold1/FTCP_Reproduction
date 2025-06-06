import os
import pandas as pd
from data import *
from model import *
from utils import *
from sampling import *

import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# ✅ Securely Retrieve API Key
mp_api_key = os.getenv("MP_API_KEY")


# ✅ Check if `FTCP_data_batteries.npy` already exists
if os.path.exists("dataframes/FTCP_data_batteries.npy") and os.path.exists("dataframes/Nsites_batteries.npy") and os.path.exists("dataframes/batteries_data_lithium.csv"):
    print("✅ Found `FTCP_data_batteries.npy` and dataframe, skipping data retrieval and FTCP representation.")
    FTCP_representation = np.load("dataframes/FTCP_data_batteries.npy")
    df_lithium = pd.read_csv("dataframes/batteries_data_lithium.csv")
    Nsites = np.load("dataframes/Nsites_batteries.npy")
else:
    print("🔍 `FTCP_data_batteries.npy` or dataframe not found, retrieving battery data from Materials Project API...")
    
    # Load the dataframe
    df_lithium = pd.read_csv("dataframes/batteries_data_lithium.csv")

    # ✅ Obtain FTCP representation
    FTCP_representation, Nsites, max_elms, max_sites = FTCP_represent(df_lithium, return_Nsites=True)

    # ✅ Save FTCP representation
    np.save("FTCP_data_batteries.npy", FTCP_representation)
    np.save("Nsites_batteries.npy", Nsites)

    print("FTCP Shape:", FTCP_representation.shape)

# Preprocess FTCP representation to obtain input X
FTCP_representation = pad(FTCP_representation, 2)
X, scaler_X = minmax(FTCP_representation)

# Get Y from filtered dataframe
dataframe = df_lithium
prop = ['fracA_discharge']
Y = dataframe[prop].values  # ✅ Now X and Y match perfectly
scaler_y = MinMaxScaler()
Y = scaler_y.fit_transform(Y)

# ✅ Split data into training and test sets
ind_train, ind_test = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=21)
X_train, X_test = X[ind_train], X[ind_test]
y_train, y_test = Y[ind_train], Y[ind_test]

# ✅ Save training data
np.save("X_train_batteries.npy", X_train)
np.save("y_train_batteries.npy", y_train)

# ✅ Get model
VAE, encoder, decoder, regression = FTCP(X_train, y_train, coeffs=(2, 10,))

# ✅ Learning rate scheduling
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, min_lr=1e-6)

def scheduler(epoch, lr):
    if epoch == 50:
        return 1e-4
    elif epoch == 100:
        return 5e-5
    return lr

schedule_lr = LearningRateScheduler(scheduler)

class TrackIndividualLosses(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, encoder, decoder, regression):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.encoder = encoder
        self.decoder = decoder
        self.regression = regression
        self.loss_history = {
            'epoch': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'property_loss': [],
            'total_loss': [],
        }

    def on_epoch_end(self, epoch, logs=None):
        idx = np.random.choice(len(self.X_train), size=256, replace=False)
        X_batch = self.X_train[idx]
        y_batch = self.y_train[idx]

        encoder_inputs, regression_inputs = X_batch, y_batch
        pred = self.model([encoder_inputs, regression_inputs], training=False)

        # Forward pass
        z_mean = self.encoder.get_layer('z_mean').output
        z_log_var = self.encoder.get_layer('z_log_var').output

        z_mean_model = tf.keras.Model(self.encoder.input, z_mean)
        z_log_var_model = tf.keras.Model(self.encoder.input, z_log_var)

        z_mean_val = z_mean_model.predict(encoder_inputs)
        z_log_var_val = z_log_var_model.predict(encoder_inputs)

        recon_loss = np.sum((encoder_inputs - pred.numpy()) ** 2)
        kl_loss = -0.5 * np.mean(1 + z_log_var_val - np.square(z_mean_val) - np.exp(z_log_var_val))
        y_hat_val = self.regression.predict(encoder_inputs)
        prop_loss = np.sum((regression_inputs[:, :y_hat_val.shape[1]] - y_hat_val) ** 2)

        total_loss = np.mean(recon_loss + 2 * kl_loss + 10 * prop_loss)

        self.loss_history['epoch'].append(epoch + 1)
        self.loss_history['reconstruction_loss'].append(recon_loss)
        self.loss_history['kl_loss'].append(kl_loss)
        self.loss_history['property_loss'].append(prop_loss)
        self.loss_history['total_loss'].append(total_loss)

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.loss_history)
        df.to_csv('individual_losses.csv', index=False)
        print("✅ Saved individual loss breakdown to 'individual_losses.csv'")


# Initialize your callback
loss_tracker = TrackIndividualLosses(X_train, y_train, encoder=encoder, decoder=decoder, regression=regression)

# ✅ Compile and Train Model
VAE.compile(optimizer=optimizers.RMSprop(learning_rate=5e-4))

VAE.fit(
    [X_train, y_train],
    X_train,
    shuffle=True,
    batch_size=256,
    epochs=200,
    callbacks=[reduce_lr, schedule_lr, loss_tracker],
)

# ✅ Save trained model
VAE.save("FTCP_VAE_batteries.keras")
encoder.save("FTCP_encoder_batteries.keras")
decoder.save("FTCP_decoder_batteries.keras")
regression.save("FTCP_regression_batteries.keras")

print("✅ Model training for batteries completed! Files saved successfully.")

#%% Visualize latent space with two arbitrary dimensions
# Predict latent space representations
train_latent = encoder.predict(X_train, verbose=1)
y_train_, y_test_ = scaler_y.inverse_transform(y_train), scaler_y.inverse_transform(y_test)

# Plot settings
font_size = 26
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size - 2
plt.rcParams['ytick.labelsize'] = font_size - 2

# Create figure with one subplot
fig, ax = plt.subplots(1, 1, figsize=(9, 7.3))

# Scatter plot for fracA_discharge
s0 = ax.scatter(train_latent[:, 0], train_latent[:, 1], s=7, c=np.squeeze(y_train_[:, 0]), cmap="viridis")
plt.colorbar(s0, ax=ax, label="fracA_discharge")
fig.text(0.016, 0.92, '(A) fracA_discharge', fontsize=font_size)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("latent_space.png")
plt.show()


#%% Evalute Reconstruction, and Target-Learning Branch Error
X_test_recon = VAE.predict([X_test, y_test], verbose=1)
X_test_recon_ = inv_minmax(X_test_recon, scaler_X)
X_test_recon_[X_test_recon_ < 0.1] = 0
X_test_ = inv_minmax(X_test, scaler_X)

# Mean absolute percentage error
def MAPE(y_true, y_pred):
    # Add a small value to avoid division of zero
    y_true, y_pred = np.array(y_true+1e-12), np.array(y_pred+1e-12)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Mean absolute error
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)

# Mean absolute error for reconstructed site coordinate matrix
def MAE_site_coor(SITE_COOR, SITE_COOR_recon, Nsites):
    site = []
    site_recon = []
    # Only consider valid sites, namely to exclude zero padded (null) sites
    for i in range(len(SITE_COOR)):
        site.append(SITE_COOR[i, :Nsites[i], :])
        site_recon.append(SITE_COOR_recon[i, :Nsites[i], :])
    site = np.vstack(site)
    site_recon = np.vstack(site_recon)
    return np.mean(np.ravel(np.abs(site - site_recon)))

# Read string of elements considered in the study (to get dimension for element matrix)
elm_str = joblib.load('data/element.pkl')
# Get lattice constants, abc
abc = X_test_[:, len(elm_str), :3]
abc_recon = X_test_recon_[:, len(elm_str), :3]
print('abc (MAPE): ', MAPE(abc,abc_recon))

# Get lattice angles, alpha, beta, and gamma
ang = X_test_[:, len(elm_str)+1, :3]
ang_recon = X_test_recon_[:, len(elm_str)+1, :3]
print('angles (MAPE): ', MAPE(ang, ang_recon))

# Get site coordinates
coor = X_test_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
coor_recon = X_test_recon_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
print('coordinates (MAE): ', MAE_site_coor(coor, coor_recon, Nsites[ind_test]))

# Get accuracy of reconstructed elements
elm_accu = []
for i in range(max_elms):
    elm = np.argmax(X_test_[:, :len(elm_str), i], axis=1)
    elm_recon = np.argmax(X_test_recon_[:, :len(elm_str), i], axis=1)
    elm_accu.append(metrics.accuracy_score(elm, elm_recon))
print(f'Accuracy for {len(elm_str)} elements are respectively: {elm_accu}')

# Get target-learning branch regression error
y_test_hat = regression.predict(X_test, verbose=1)
y_test_hat_ = scaler_y.inverse_transform(y_test_hat)
print(f'The regression MAE for {prop} are respectively', MAE(y_test_, y_test_hat_))

#%% Sampling the latent space and perform inverse design

# Specify design targets, e.g., high voltage and energy density
# 1) Pick Nsamples seeds closest to your target property
target_frac = 0.8
Nsamples    = 10
distances   = np.abs(y_train_.flatten() - target_frac)
ind_seed    = np.argsort(distances)[:Nsamples]
seeds       = train_latent[ind_seed]  # shape (Nsamples, latent_dim)

# 2) Define local perturbation parameters
Nperturb = 3      # how many perturbed copies per seed
Lp_scale = 0.6    # Gaussian noise σ in latent space

# 3) Tile seeds and add noise
samples_lp = np.repeat(seeds, Nperturb, axis=0)           # shape (Nsamples*Nperturb, latent_dim)
noise      = np.random.normal(0, Lp_scale, samples_lp.shape)
samples_lp += noise

# 4) Decode & inverse‑scale back to FTCP representation
ftcp_lp = decoder.predict(samples_lp, verbose=1)
ftcp_lp = inv_minmax(ftcp_lp, scaler_X)

# 5) Convert FTCP → chemistry & write CIFs
elm_str = joblib.load('data/element.pkl')
pred_formulas_lp, pred_abc_lp, pred_ang_lp, pred_latt_lp, pred_site_coor_lp, ind_unique_lp = get_info(
    ftcp_lp,
    max_elms, max_sites,
    elm_str=elm_str,
    to_CIF=True,
    check_uniqueness=True,
    mp_api_key=mp_api_key

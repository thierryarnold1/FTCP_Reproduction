import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Conv1D, Conv2DTranspose, \
    LeakyReLU, Activation, Flatten, Reshape, BatchNormalization, Cropping1D, Dropout
from tensorflow.keras.models import Model
from config import FTCP_CONFIG

# --- Custom KL Divergence Layer ---
class KLDivergenceLayer(tf.keras.layers.Layer):
    def __init__(self, coeff, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.coeff = coeff

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        self.add_loss(self.coeff * kl_loss)
        return z_mean

# --- Custom Property Loss Layer ---
class PropertyLossLayer(tf.keras.layers.Layer):
    def __init__(self, coeff, prop_dim, **kwargs):
        super(PropertyLossLayer, self).__init__(**kwargs)
        self.coeff = coeff
        self.prop_dim = prop_dim

    def call(self, inputs):
        reg_in, y_hat_val = inputs
        loss = K.sum(K.square(reg_in[:, :self.prop_dim] - y_hat_val))
        self.add_loss(self.coeff * loss)
        return y_hat_val

# --- Custom Semi-supervised Property Loss Layer ---
class SemiPropertyLossLayer(tf.keras.layers.Layer):
    def __init__(self, coeff, prop_dim, semi_prop_dim, **kwargs):
        super(SemiPropertyLossLayer, self).__init__(**kwargs)
        self.coeff = coeff
        self.prop_dim = prop_dim
        self.semi_prop_dim = semi_prop_dim

    def call(self, inputs):
        y_semi, y_semi_hat_val = inputs
        loss = K.sum(K.square(y_semi_hat_val - y_semi[:, self.prop_dim:self.prop_dim+self.semi_prop_dim]))
        self.add_loss(self.coeff * loss)
        return y_semi_hat_val

# --- Updated FTCP model using config ---
def FTCP(X_train, y_train, config=FTCP_CONFIG, semi=False, label_ind=None, prop_dim=None):
    K.clear_session()

    latent_dim = config["latent_dim"]
    filters = config["filters"]
    filter_sizes = config["filter_sizes"]
    strides = config["strides"]
    coeff_KL = config["coeff_KL"]
    coeff_prop = config["coeff_prop"]
    activation_dense = config["activation_dense"]
    activation_latent = config["activation_latent"]
    activation_conv = config["activation_conv"]
    activation_final_decoder = config["activation_final_decoder"]
    regression_output_activation = config["regression_output_activation"]
    regression_hidden = config["regression_hidden"]
    dense_latent_dim = config["dense_latent_dim"]
    use_batchnorm = config["use_batchnorm"]
    use_dropout = config["use_dropout"]
    dropout_rate = config["dropout_rate"]
    leakyrelu_alpha = config["leakyrelu_alpha"]

    if semi:
        coeff_prop_semi = config["coeff_prop_semi"]
        assert tuple(label_ind) != None, "You must input the index for semi-supervised property to do semi-supervised learning"
        assert prop_dim != None, "You must input the dimensions of the properties to do semi-supervised learning"
        prop_dim, semi_prop_dim = prop_dim

    input_dim = X_train.shape[1]
    channel_dim = X_train.shape[2]
    regression_dim = y_train.shape[1]

    encoder_inputs = Input(shape=(input_dim, channel_dim,))
    regression_inputs = Input(shape=(regression_dim,))

    # Encoder
    x = encoder_inputs
    for f, k, s in zip(filters, filter_sizes, strides):
        x = Conv1D(f, k, strides=s, padding='same')(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=leakyrelu_alpha)(x)
    x = Flatten()(x)
    x = Dense(dense_latent_dim, activation=activation_latent)(x)
    z_mean = Dense(latent_dim, activation='linear')(x)
    z_log_var = Dense(latent_dim, activation='linear')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean + K.exp(z_log_var/2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    _ = KLDivergenceLayer(coeff_KL)([z_mean, z_log_var])

    encoder = Model(encoder_inputs, z, name='encoder')

    # Regression branch
    def regression_block(source):
        x = Activation(activation_dense)(source)
        for units in regression_hidden:
            x = Dense(units, activation=activation_dense)(x)
            if use_dropout:
                x = Dropout(dropout_rate)(x)
        return x

    if not semi:
        x_reg = regression_block(z_mean)
        y_hat = Dense(regression_dim, activation=regression_output_activation)(x_reg)
        y_hat = PropertyLossLayer(coeff_prop, prop_dim=regression_dim)([regression_inputs, y_hat])
        regression = Model(encoder_inputs, y_hat, name='target-learning branch')
    else:
        x_reg = regression_block(z_mean)
        y_hat = Dense(prop_dim, activation=regression_output_activation)(x_reg)
        y_hat = PropertyLossLayer(coeff_prop, prop_dim)([regression_inputs, y_hat])

        x_semi = regression_block(z_mean)
        y_semi_hat = Dense(semi_prop_dim, activation=regression_output_activation)(x_semi)
        regression = Model(encoder_inputs, [y_hat, y_semi_hat], name='target-learning branch')

        y_semi = Lambda(lambda x: tf.gather(x, tf.where(tf.reduce_any(tf.equal(x[:, -1:], tf.convert_to_tensor(label_ind[:, None])), axis=1))[:,0], axis=0))(regression_inputs)
        y_semi_hat = Lambda(lambda x: tf.gather(x, tf.where(tf.reduce_any(tf.equal(x[:, -1:], tf.convert_to_tensor(label_ind[:, None])), axis=1))[:,0], axis=0))(y_semi_hat)
        _ = SemiPropertyLossLayer(coeff_prop_semi, prop_dim, semi_prop_dim)([y_semi, y_semi_hat])

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    map_size = K.int_shape(encoder.layers[-6].output)[1]
    x_dec = Dense(filters[-1] * map_size, activation=activation_dense)(latent_inputs)
    x_dec = Reshape((map_size, 1, filters[-1]))(x_dec)
    if use_batchnorm:
        x_dec = BatchNormalization()(x_dec)

    for f, k, s in zip(filters[::-1][1:], filter_sizes[::-1][1:], strides[::-1][1:]):
        x_dec = Conv2DTranspose(f, (k, 1), strides=(s, 1), padding='same')(x_dec)
        if use_batchnorm:
            x_dec = BatchNormalization()(x_dec)
        x_dec = Activation(activation_conv)(x_dec)

    x_dec = Conv2DTranspose(channel_dim, (filter_sizes[0], 1), strides=(strides[0], 1), padding='same')(x_dec)
    x_dec = Activation(activation_final_decoder)(x_dec)
    decoder_outputs = Lambda(lambda x: K.squeeze(x, axis=2))(x_dec)

    current_length = K.int_shape(decoder_outputs)[1]
    crop_amount = (current_length - input_dim) // 2

    if crop_amount > 0:
        decoder_outputs = Cropping1D(cropping=(crop_amount, crop_amount))(decoder_outputs)

    decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    reconstructed_outputs = decoder(z)
    VAE = Model(inputs=[encoder_inputs, regression_inputs], outputs=reconstructed_outputs)

    def vae_loss(y_true, y_pred):
        loss_recon = K.sum(K.square(y_true - y_pred))
        return K.mean(loss_recon)

    return VAE, encoder, decoder, regression, vae_loss

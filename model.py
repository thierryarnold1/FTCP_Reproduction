import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Conv1D, Conv2DTranspose, \
    LeakyReLU, Activation, Flatten, Reshape, BatchNormalization
from tensorflow.keras.models import Model

# --- Custom KL Divergence Layer ---
class KLDivergenceLayer(tf.keras.layers.Layer):
    def __init__(self, coeff, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.coeff = coeff

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        self.add_loss(self.coeff * kl_loss)
        # Pass through one of the inputs unchanged.
        return z_mean

# --- Custom Property Loss Layer ---
class PropertyLossLayer(tf.keras.layers.Layer):
    def __init__(self, coeff, prop_dim, **kwargs):
        super(PropertyLossLayer, self).__init__(**kwargs)
        self.coeff = coeff
        self.prop_dim = prop_dim

    def call(self, inputs):
        reg_in, y_hat_val = inputs
        # Compute property loss on the first 'prop_dim' elements.
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

def FTCP(X_train, y_train, coeffs=(2, 10,), semi=False, label_ind=None, prop_dim=None):
    
    K.clear_session()
    
    if not semi:
        coeff_KL, coeff_prop = coeffs
    else:
        coeff_KL, coeff_prop, coeff_prop_semi = coeffs
    
    latent_dim = 256
    max_filters = 128
    filter_size = [5, 3, 3]
    strides = [2, 2, 1]
    
    input_dim = X_train.shape[1]
    channel_dim = X_train.shape[2]
    regression_dim = y_train.shape[1]
    
    encoder_inputs = Input(shape=(input_dim, channel_dim,))
    regression_inputs = Input(shape=(regression_dim,))
    
    if semi:
        assert tuple(label_ind) != None, "You must input the index for semi-supervised property to do semi-supervised learning"
        assert prop_dim != None, "You must input the dimensions of the properties to do semi-supervised learning"
        prop_dim, semi_prop_dim = prop_dim
        
        label_ind = tf.convert_to_tensor(label_ind, dtype=tf.int64)
        def get_idn(y):
            y_ind = y[:, -1]  
            y_ind = tf.dtypes.cast(y_ind, tf.int64)
            com_ind = tf.sets.intersection(y_ind[None, :], label_ind[None, :])
            com_ind = tf.sparse.to_dense(com_ind)
            com_ind = tf.squeeze(com_ind)
            com_ind = tf.reshape(com_ind, (tf.shape(com_ind)[0], 1))
            semi_ind = tf.where(tf.equal(y_ind, com_ind))[:, -1]
            return semi_ind
        semi_ind = Lambda(get_idn)(regression_inputs)
        
    # Encoder
    x = Conv1D(max_filters//4, filter_size[0], strides=strides[0], padding='same')(encoder_inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(max_filters//2, filter_size[1], strides=strides[1], padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(max_filters, filter_size[2], strides=strides[2], padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='sigmoid')(x)
    z_mean = Dense(latent_dim, activation='linear')(x)
    z_log_var = Dense(latent_dim, activation='linear')(x)
    
    # Reparameterization
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean + K.exp(z_log_var/2) * epsilon
    
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # Wrap KL divergence in a custom layer.
    _ = KLDivergenceLayer(coeff_KL)([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, z, name='encoder')
    
    # Regression branch (for property prediction)
    if not semi:
        x_reg = Activation('relu')(z_mean)
        x_reg = Dense(128, activation="relu")(x_reg)
        x_reg = Dense(32, activation="relu")(x_reg)
        y_hat = Dense(regression_dim, activation='sigmoid')(x_reg)
        # Wrap property loss computation in a custom layer.
        y_hat = PropertyLossLayer(coeff_prop, prop_dim)([regression_inputs, y_hat])
        regression = Model(encoder_inputs, y_hat, name='target-learning branch')
    else:
        x_reg = Activation('relu')(z_mean)
        x_reg = Dense(128, activation="relu")(x_reg)
        x_reg = Dense(32, activation="relu")(x_reg)
        y_hat = Dense(prop_dim, activation='sigmoid')(x_reg)
        # Add property loss for the supervised part.
        y_hat = PropertyLossLayer(coeff_prop, prop_dim)([regression_inputs, y_hat])
        
        x_semi = Activation('relu')(z_mean)
        x_semi = Dense(128, activation="relu")(x_semi)
        x_semi = Dense(32, activation="relu")(x_semi)
        y_semi_hat = Dense(semi_prop_dim, activation='sigmoid')(x_semi)
        regression = Model(encoder_inputs, [y_hat, y_semi_hat], name='target-learning branch')
        
        # Gather semi-supervised indices.
        y_semi = Lambda(lambda x: tf.gather(x, semi_ind, axis=0))(regression_inputs)
        y_semi_hat = Lambda(lambda x: tf.gather(x, semi_ind, axis=0))(y_semi_hat)
        # Wrap semi-supervised property loss in a custom layer.
        _ = SemiPropertyLossLayer(coeff_prop_semi, prop_dim, semi_prop_dim)([y_semi, y_semi_hat])
    
    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    map_size = K.int_shape(encoder.layers[-6].output)[1]
    x_dec = Dense(max_filters * map_size, activation='relu')(latent_inputs)
    x_dec = Reshape((map_size, 1, max_filters))(x_dec)
    x_dec = BatchNormalization()(x_dec)
    x_dec = Conv2DTranspose(max_filters//2, (filter_size[2], 1), strides=(strides[2], 1), padding='same')(x_dec)
    x_dec = BatchNormalization()(x_dec)
    x_dec = Activation('relu')(x_dec)
    x_dec = Conv2DTranspose(max_filters//4, (filter_size[1], 1), strides=(strides[1], 1), padding='same')(x_dec)
    x_dec = BatchNormalization()(x_dec)
    x_dec = Activation('relu')(x_dec)
    x_dec = Conv2DTranspose(channel_dim, (filter_size[0], 1), strides=(strides[0], 1), padding='same')(x_dec)
    x_dec = Activation('sigmoid')(x_dec)
    decoder_outputs = Lambda(lambda x: K.squeeze(x, axis=2))(x_dec)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    
    reconstructed_outputs = decoder(z)
    VAE = Model(inputs=[encoder_inputs, regression_inputs], outputs=reconstructed_outputs)
    
    VAE.summary()
    
    # Custom loss now only computes the reconstruction loss.
    def vae_loss(y_true, y_pred):
        loss_recon = K.sum(K.square(y_true - y_pred))
        return K.mean(loss_recon)
    
    return VAE, encoder, decoder, regression, vae_loss

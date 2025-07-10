

import horovod.tensorflow as hvd
import tensorflow as tf
import numpy as np
import os

# Initialize Horovod
hvd.init()

# Pin GPU to local rank
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)

# Dummy time series data
def generate_data(seq_len=50, num_samples=10000):
    X = np.random.rand(num_samples, seq_len, 1)
    y = np.random.randint(0, 2, size=(num_samples, 1))
    return X, y

X_train, y_train = generate_data()

# Simple LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Optimizer with Horovod distributed wrapper
opt = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Callbacks
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    tf.keras.callbacks.ModelCheckpoint(f'checkpoints/model_rank_{hvd.rank()}.h5', save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=f'logs/rank_{hvd.rank()}')
]

# Fit model
model.fit(X_train, y_train,
          batch_size=128,
          epochs=10,
          callbacks=callbacks,
          verbose=1 if hvd.rank() == 0 else 0)

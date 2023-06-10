import tensorflow as tf
from datetime import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Disable all GPUS, TF Metal is very slow
# https://stackoverflow.com/questions/70653251/why-gpu-is-3-5-times-slower-than-the-cpu-on-apple-m1-mac
tf.config.set_visible_devices([], 'GPU')


start = datetime.now()

# with tf.device('/CPU:0'):
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())


model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

end = datetime.now()
print("Time taken:", end - start)
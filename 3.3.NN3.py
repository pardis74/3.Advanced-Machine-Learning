#At first I import the necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds

# In this part we load CIFAR-10 dataset
(train_ds, test_ds), info = tfds.load(name='cifar10', split=['train', 'test'], with_info=True, as_supervised=True)

# Normalize pixel values to between 0 and 1 for the DenseNet
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

# In this part we add data augmentation to the training dataset
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    tf.keras.layers.experimental.preprocessing.RandomCrop(32, 32),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.map(normalize_img)
test_ds = test_ds.map(normalize_img)

# Prepare datasets for training and testing
BATCH_SIZE = 32
train_ds = train_ds.batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

# In this part we define the DenseNet model
densenet = tf.keras.applications.DenseNet121(
    input_shape=(32, 32, 3),
    weights='imagenet',
    include_top=False,
    pooling='max'
)
model = tf.keras.Sequential([
    densenet,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model for show the results in output
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# In this part I trained the data and put it 100 epochs but it took lots of time to get the result
model.fit(train_ds, epochs=100, validation_data=test_ds)

# Evaluate the model on test data (Show the results by printing)
test_loss, test_acc = model.evaluate(test_ds)
print('Test accuracy:', test_acc)
from imports import *
from data import *

BATCH_SIZE = 64
IMG_SIZE = (224, 224)
NUM_CLASSES = 39
EPOCHS = 10

train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))



base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze for transfer learning

inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)



model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


steps_per_epoch = math.ceil(len(train_ds))

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=3e-3,
    decay_steps=steps_per_epoch * EPOCHS,
    end_learning_rate=1e-5
)

model.optimizer.learning_rate = lr_schedule



checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

tensorboard_cb = keras.callbacks.TensorBoard(log_dir="logs")



history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, tensorboard_cb]
)



test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

print("Training balanced model...")

train_dir = 'archive/chest_xray/train'
test_dir = 'archive/chest_xray/test'

img_size = (150, 150)
batch_size = 32

# Calculate class weights for imbalance
train_datagen_temp = ImageDataGenerator(rescale=1./255)
temp_gen = train_datagen_temp.flow_from_directory(train_dir, target_size=img_size, batch_size=1, class_mode='binary')
normal_count = np.sum(temp_gen.classes == 0)
pneumonia_count = np.sum(temp_gen.classes == 1)
total = normal_count + pneumonia_count
class_weight = {0: total/(2*normal_count), 1: total/(2*pneumonia_count)}
print(f"Class weights: Normal={class_weight[0]:.2f}, Pneumonia={class_weight[1]:.2f}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers[:-15]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint('best_pneumonia_model.h5', monitor='accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

print("\nTraining with class weights...\n")

history = model.fit(
    train_generator,
    epochs=15,
    class_weight=class_weight,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

model = tf.keras.models.load_model('best_pneumonia_model.h5')
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

predictions = model.predict(test_generator)
pred_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=['NORMAL', 'PNEUMONIA']))
print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, pred_classes))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], linewidth=2)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], linewidth=2)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("\n✓ Best model saved as 'best_pneumonia_model.h5'")

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# %% Step 0 : Find GPU

print("--- Finding GPU ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Availiable GPU: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("GPU NOT FOUND. CPU WILL BE USED")
print("---------------------")

# %% Step 1 : Data Processing

# 1. Define Variable
IMG_WIDTH, IMG_HEIGHT = 500, 500
BATCH_SIZE = 8     # 24 should be used but because of the GPU limitation 8 was used

# 2. Define the Data location
train_dir = 'Data/train'
validation_dir = 'Data/valid'

# 3. Training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# 4. Re-scaling validation data
validation_datagen = ImageDataGenerator(rescale=1./255) # To match step5

# 5. Call data from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Data/train
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, # Data/valid
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Class Indices
print("Found Class:", train_generator.class_indices)

# %% Step 2 & 3 : NN Design and Hyperparameter analysis

# Model A
model_A = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'), 
    layers.Dropout(0.2), 

    # Fianl 3 layers using 'softmax'
    layers.Dense(3, activation='softmax') 
])

# Model B
model_B = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'), # More Neuran
    layers.Dropout(0.3), 
    layers.Dense(3, activation='softmax')
])

# Model A Compile
model_A.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model B Compile
model_B.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Model Architecture Summary
print("--- Model A Architecture ---")
model_A.summary()
print("\n--- Model B Architecture ---")
model_B.summary()

# %% Step 4 : Model Trinaing & Evaluation

# EarlyStopping : To prvent overfitting
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# --- Model A Training ---
print("\n--- Model A Training ---")
history_A = model_A.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[early_stop]
)
# Save Model A into h5 format
model_A.save('model_A.h5') 

# --- Model B Training ---
print("\n--- Model B Training ---")
history_B = model_B.fit(
    train_generator,
    epochs=15, 
    validation_data=validation_generator,
    callbacks=[early_stop]
)
# Save Model B into h5 format
model_B.save('model_B.h5')


# --- Visualization ---

# Model A visualization and saving
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_A.history['accuracy'], label='Train Acc')
plt.plot(history_A.history['val_accuracy'], label='Val Acc')
plt.title('Model A - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_A.history['loss'], label='Train Loss')
plt.plot(history_A.history['val_loss'], label='Val Loss')
plt.title('Model A - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('model_A_performance.png')
plt.show()


# Model B visualization and saving
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_B.history['accuracy'], label='Train Acc')
plt.plot(history_B.history['val_accuracy'], label='Val Acc')
plt.title('Model B - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_B.history['loss'], label='Train Loss')
plt.plot(history_B.history['val_loss'], label='Val Loss')
plt.title('Model B - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('model_B_performance.png')
plt.show()

print("Project2 step 1 - 4 is DONE!!!!!!!!")
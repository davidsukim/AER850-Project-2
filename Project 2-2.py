import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf

# %% Step 0 : Find GPU

print("--- Finding GPU ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Available GPU: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("GPU NOT FOUND. CPU WILL BE USED")
print("---------------------")

# %% Step 5 : Model Testing

# 1. Define Variable
IMG_WIDTH, IMG_HEIGHT = 500, 500 # match with the training data
class_labels = ['crack', 'missing-head', 'paint-off']

# 2. Test image list
base_dir = os.getcwd()
test_image_paths = [
    os.path.join(base_dir, 'Data', 'test', 'crack', 'test_crack.jpg'),
    os.path.join(base_dir, 'Data', 'test', 'missing-head', 'test_missinghead.jpg'),
    os.path.join(base_dir, 'Data', 'test', 'paint-off', 'test_paintoff.jpg')
]

# List of models to test
model_files = ['model_A.h5', 'model_B.h5']

# Loop through each model
for model_file in model_files:
    print(f"\n--- Testing {model_file} ---")
    
    # Load the current model
    try:
        model = load_model(model_file)
        print(f"'{model_file}' Loaded Model")
    except IOError:
        print(f"ERROR: '{model_file}' File cannot be found")
        continue # Skip to next model if not found

    # Model name without extension for saving files
    model_name = os.path.splitext(model_file)[0]

    # 4. Image Prediction for the current model
    for img_path in test_image_paths:
        if not os.path.exists(img_path):
            print(f"ERROR: {img_path} COULDN'T FIND THE FILE")
            continue

        # Load image
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)

        pred_index = np.argmax(prediction[0])
        pred_label = class_labels[pred_index]

        true_label = img_path.split(os.sep)[-2]

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        
        # Title (Updated to show which model is predicting)
        title_text = f"[{model_name}] True Label: {true_label}\n"
        title_text += f"Predicted Label: {pred_label}\n\n"
        
        # Probability
        prob_text = ""
        for i in range(len(class_labels)):
            prob_text += f"{class_labels[i].title()}: {prediction[0][i]*100:.1f}%\n"
            
        plt.title(title_text)
        
        # Display the result on the image
        plt.text(10, IMG_HEIGHT - 60, prob_text, 
                 color='green', 
                 fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, pad=0.5))
                 
        plt.axis('off')
        
        # Save Test Result with model name prefix
        # e.g., model_A_test_crack_prediction.png
        original_filename = os.path.basename(img_path).replace('.jpg', '_prediction.png')
        save_name = f"{model_name}_{original_filename}"
        
        plt.savefig(save_name)
        print(f"'{save_name}' SAVED.")
        plt.show()

print("\n--- ALL MODEL TESTS FINISHED!! ---")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf


# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_TOP = 5
EPOCHS_FINE = 10


# 1. Verify dataset structure
print("Current directory contents:", os.listdir())




# 2. Configure paths with validation
def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    return path
train_dir = validate_path("/kaggle/input/skinlesions-dataset/SkinLesions_Dataset/Training_dataset_SkinLesions")
test_dir = validate_path("/kaggle/input/skinlesions-dataset/SkinLesions_Dataset/Test_dataset_SkinLesions")    


train_df = pd.read_csv('/kaggle/input/skinlesions-dataset/SkinLesions_Dataset/SkinLesions_train.csv')

# 4. Dynamic path builder with extension checking
def build_image_path(row_id, base_dir):
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(base_dir, f"{row_id}{ext}")
        if os.path.exists(path):
            return path
    return None  # For missing files handling

# Create filename column
train_df['filename'] = train_df['ID'].apply(
    lambda x: build_image_path(x, train_dir)
)
train_df['Target']=train_df['Target'].astype(str)


train_df = pd.read_csv('/kaggle/input/skinlesions-dataset/SkinLesions_Dataset/SkinLesions_train.csv')

# 4. Dynamic path builder with extension checking
def build_image_path(row_id, base_dir):
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(base_dir, f"{row_id}{ext}")
        if os.path.exists(path):
            return path
    return None  # For missing files handling

# Create filename column
train_df['filename'] = train_df['ID'].apply(
    lambda x: build_image_path(x, train_dir)
)
train_df['Target']=train_df['Target'].astype(str)


# 5. Handle missing files
print(f"Missing {train_df['filename'].isna().sum()} training images")
train_df = train_df.dropna(subset=['filename'])


# 6. Class weights for imbalance
class_weights = class_weight.compute_class_weight('balanced',
                                               classes=np.unique(train_df['Target']),
                                               y=train_df['Target'])
class_weights_dict = dict(enumerate(class_weights))


# 7. Split data with verified paths
train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['Target'],
    random_state=42
)


# Data generators with preprocessing
# 8. Enhanced data generators
from tensorflow.keras.applications.efficientnet import preprocess_input

def create_generator(df, datagen, is_train=True):
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='Target',  # Ensure this matches CSV column name exactly
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=is_train,
        validate_filenames=True
    )

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# 9. Verify generators
print("\nTraining generator:")
train_generator = create_generator(train_data, train_datagen)
print("\nValidation generator:")
val_generator = create_generator(val_data, val_datagen, is_train=False)

test_files = []
for fname in os.listdir(test_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        test_files.append(fname)

test_df = pd.DataFrame({
    'ID': [os.path.splitext(f)[0] for f in test_files],
    'filename': [os.path.join(test_dir, f) for f in test_files]
})

print(f"\nFound {len(test_df)} test images")

# Test generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    class_mode=None,
    shuffle=False,
    target_size=IMG_SIZE,
      batch_size=BATCH_SIZE
)


# Build the model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# Freeze base model layers
base_model.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_model_top.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1
    )
]



# Train top layers
history_top = model.fit(
    train_generator,
    epochs=EPOCHS_TOP,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# Fine-tune the model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Callbacks for fine-tuning
callbacks_fine = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model_top.keras', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_FINE,
    validation_data=val_generator,
    callbacks=callbacks_fine,
    class_weight=class_weights_dict
)



# Evaluate on validation data
val_preds = model.predict(val_generator)
val_preds_classes = np.argmax(val_preds, axis=1) + 1  # Convert back to 1-8

val_true = val_data['TARGET'].values
balanced_acc = balanced_accuracy_score(val_true, val_preds_classes)
print(f"Balanced Accuracy on Validation: {balanced_acc:.4f}")

# Prepare test data
test_dir = 'Test_dataset_SkinLesions'
test_filenames = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
test_ids = [os.path.splitext(f)[0] for f in test_filenames]

test_df = pd.DataFrame({'ID': test_ids})
test_df['filename'] = test_dir + '/' + test_df['ID'] + '.jpg'

# Test generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)
# Predict on test set
test_preds = model.predict(test_generator)
test_preds_classes = np.argmax(test_preds, axis=1) + 1  # Convert to 1-8



# Create submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'TARGET': test_preds_classes})
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully.")
    

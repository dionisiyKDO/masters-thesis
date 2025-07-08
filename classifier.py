import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adamax, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)


class Classifier:
    
    AVAILABLE_MODELS = [
        'OwnV2', 'OwnV1', 'SimpleNet', 'VGG16', 'VGG19', 'AlexNet', 
        'InceptionV3', 'EfficientNetV2', 'ResNet50', 'InceptionResNetV2'
    ]
    
    def __init__(self, 
                 model_name: str = 'OwnV2',
                 img_size: Tuple[int, int] = (150, 150),
                 data_dir: Optional[str] = None,
                 class_labels: Optional[Dict[str, int]] = None):

        self.model_name = model_name
        self.img_size = img_size
        self.img_shape = (*img_size, 3)
        self.data_dir = data_dir
        
        # Auto-detect classes from directory structure or use provided labels
        self.class_labels = class_labels or self._detect_classes()
        self.num_classes = len(self.class_labels)
        
        self.model = None
        self.history = None
        self.train_generator = None
        self.test_generator = None
        
        self._setup_environment()

    def _setup_environment(self):
        """Configure GPU settings and optimize TensorFlow."""
        # Configure GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except RuntimeError:
                pass  # Memory growth must be set before GPUs have been initialized
    
        # Silence TensorFlow logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            # 0 = all messages are logged (default behavior)
            # 1 = INFO messages are not printed
            # 2 = INFO and WARNING messages are not printed
            # 3 = INFO, WARNING, and ERROR messages are not printed

    def _detect_classes(self) -> Dict[str, int]:
        """Auto-detect class labels from directory structure."""
        if not self.data_dir:
            print("No data directory")
            exit()
        
        train_dir = Path(self.data_dir) / 'train'
        if not train_dir.exists():
            train_dir = Path(self.data_dir)
        
        if train_dir.exists():
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            return {cls: idx for idx, cls in enumerate(classes)}
        
        return {}


    def setup_data_generators(self, 
                            train_dir: str,
                            test_dir: str,
                            batch_size: int = 32,
                            seed: int = 42) -> Tuple[int, int]:
        """
        Setup data generators for training and validation.
        
        Args:
            train_dir: Path to training data directory
            test_dir: Path to test data directory (optional)
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation if no test_dir
            augmentation_config: Custom augmentation parameters
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (steps_per_epoch, validation_steps)
        """
        # Default augmentation config
        default_augmentation = {
            'rotation_range': 10,
            'brightness_range': (0.9, 1.1),
            'width_shift_range': 0.005,
            'height_shift_range': 0.005,
            'shear_range': 10,
            'horizontal_flip': True,
        }
        
        # Create data generators
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **default_augmentation
        )
        
        # Setup generators
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=seed
        )
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            seed=seed,
        )
        
        # Update class labels from generator
        self.class_labels = {v: k for k, v in self.train_generator.class_indices.items()}
        self.num_classes = len(self.class_labels)
        
        # steps_per_epoch = self.train_generator.samples // batch_size
        # validation_steps = self.test_generator.samples // batch_size
        
        steps_per_epoch = ceil(self.train_generator.samples / batch_size)
        validation_steps = ceil(self.test_generator.samples / batch_size)
        
        return steps_per_epoch, validation_steps

    def build_model(self) -> bool:
        """Build the specified model architecture."""
        try:
            if self.model_name == 'SimpleNet':
                self.model = Sequential([
                    Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', input_shape=self.img_shape),
                    MaxPooling2D(pool_size=2),
                    Conv2D(filters=32, kernel_size=3, activation='relu'),
                    MaxPooling2D(pool_size=2),
                    Flatten(),
                    Dense(64, activation='relu'),

                    Dense(self.num_classes, activation='softmax')
                ])
            
            elif self.model_name == 'OwnV1':
                self.model = Sequential([
                    Conv2D(256, (12,12), strides=(5,5), activation='relu', input_shape=self.img_shape),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Conv2D(512, (6,6), strides=(2,2), activation='relu', padding='same'),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Conv2D(256, (3,3), activation='relu', padding='same'),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Flatten(),
                    Dense(4096, activation='relu'),
                    Dropout(0.5),
                    Dense(self.num_classes, activation='softmax')
                ])
            
            elif self.model_name == 'OwnV2':
                self.model = Sequential([
                    Conv2D(64, (12,12), strides=(5,5), activation='relu', input_shape=self.img_shape),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Conv2D(128, (6,6), strides=(2,2), activation='relu', padding='same'),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Conv2D(256, (3,3), activation='relu', padding='same'),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Flatten(),
                    Dense(1024, activation='relu'),
                    Dropout(0.5),
                    Dense(self.num_classes, activation='softmax')
                ])
            
            elif self.model_name == 'AlexNet':
                self.model = Sequential([
                    Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=self.img_shape),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Conv2D(256, (5,5), activation='relu', padding='same'),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Conv2D(384, (3,3), activation='relu', padding='same'),
                    Conv2D(384, (3,3), activation='relu', padding='same'),
                    Conv2D(256, (3,3), activation='relu', padding='same'),
                    MaxPooling2D((3,3), strides=(2,2)),
                    BatchNormalization(),
                    Flatten(),
                    Dense(4096, activation='relu'),
                    Dropout(0.5),
                    Dense(4096, activation='relu'),
                    Dropout(0.5),
                    Dense(self.num_classes, activation='softmax')
                ])
            
            # Transfer learning models
            else:
                base_models = {
                    'VGG16': tf.keras.applications.VGG16,
                    'VGG19': tf.keras.applications.VGG19,
                    'ResNet50': tf.keras.applications.ResNet50,
                    'InceptionV3': tf.keras.applications.InceptionV3,
                    'EfficientNetV2': tf.keras.applications.EfficientNetV2B0,
                    'InceptionResNetV2': tf.keras.applications.InceptionResNetV2
                }
                
                if self.model_name in base_models:
                    base_model = base_models[self.model_name](
                        weights='imagenet',
                        include_top=False,
                        input_shape=self.img_shape
                    )
                    base_model.trainable = False
                    
                    self.model = Sequential([
                        base_model,
                        GlobalAveragePooling2D(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(self.num_classes, activation='softmax')
                    ])
                else:
                    print(f"Unknown model: {self.model_name}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error building model: {e}")
            return False
    
    def load_model_from_checkpoint(self, checkpoint_path: str):
        """
        Load model from a checkpoint file.

        Parameters:
        - checkpoint_path (str): Path to the checkpoint file.
        """

        self._get_model()
        self.model.load_weights(checkpoint_path)

    def classify_image(self, image_path: str):
        """
        Classify an image using a trained model.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - str: Predicted class label.
        """
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_shape)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0 # normalize

        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = [label for label, index in self.class_labels.items() if index == predicted_class_index][0]

        return predicted_class_label


    def train(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epochs=20, batch_size=16, logging=True, model_summary=True, earlystop=True):
        self._get_model()
        if self.model == None: 
            print('=== Incorrect network name ===')
            return

        start_time = time.time()

        seed = 1
        train_dir = './data/Training/'
        test_dir  = './data/Testing/'
        
        # Getting data
        # region
        test_datagen =  ImageDataGenerator(rescale=1./255)
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=10,
                                           brightness_range=(0.9, 1.10),
                                           width_shift_range=0.005,
                                           height_shift_range=0.005,
                                           shear_range=10,
                                           horizontal_flip=True,
                                           )
        
        self.test_generator  =  test_datagen.flow_from_directory(test_dir,
                                                                target_size=self.img_size,
                                                                batch_size=batch_size,
                                                                class_mode="categorical",
                                                                shuffle=False,
                                                                seed=seed)
        self.train_generator = train_datagen.flow_from_directory(train_dir,
                                                                target_size=self.img_size,
                                                                batch_size=batch_size,
                                                                class_mode="categorical",
                                                                seed=seed
                                                                )
        
        steps_per_epoch =  self.train_generator.samples // batch_size
        validation_steps = self.test_generator.samples  // batch_size
        # endregion

        # Initialize model
        # region
        if model_summary: self.model.summary()
        
        self.model.compile(Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2), loss= 'categorical_crossentropy', metrics=['acc'])
        # model.compile(Adamax(learning_rate=learning_rate), loss= 'categorical_crossentropy', metrics=['acc'])
        # model.compile(SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics= ['acc'])
        
        # endregion
       
        # Network training
        # region
        if earlystop: model_es = EarlyStopping(monitor='val_acc', min_delta=1e-9, patience=8, verbose=True)
        model_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=True)
        model_cp = ModelCheckpoint(
            filepath='./checkpoints/model.epoch{epoch:02d}-val_acc{val_acc:.4f}.hdf5', 
            monitor='val_acc', 
            save_freq='epoch', 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=True, )
        
        callbacks = [model_rlr, model_cp, model_es] if earlystop else [model_rlr, model_cp]

        def on_epoch_end(epoch, val_acc):
            globals.progress = int((epoch + 1) / epochs * 100)

        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end))

        self.history = self.model.fit(self.train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      validation_data=self.test_generator,
                                      validation_steps=validation_steps,
                                      verbose=logging, 
                                      callbacks=callbacks)
        # endregion
        
        # Network evaluation + results of trining
        # region
        loss, accuracy = self.model.evaluate(self.test_generator, steps=validation_steps)
        self.training_time = time.time() - start_time

        print(f'Test loss: {loss:.4f}')
        print(f'Test accuracy: {accuracy:.4f}')
        print(f'Training time: {self.training_time:.4f}')
        return {'loss': loss, 'accuracy': accuracy, 'training_time': self.training_time}
        # endregion


    def plot_training_history(self):
        if self.history == None: 
            print('=== No history to plot ===')
            return
        history = self.history.history
        acc_epochs = pd.DataFrame({'train': history['acc'], 'val': history['val_acc']})
        loss_epochs = pd.DataFrame({'train': history['loss'], 'val': history['val_loss']})

        px.line(acc_epochs, x=acc_epochs.index, y=acc_epochs.columns[0::], title=f'Training and Evaluation Accuracy every Epoch for "{self.network_name}"', markers=True).show()
        px.line(loss_epochs, x=loss_epochs.index, y=loss_epochs.columns[0::], title=f'Training and Evaluation Loss every Epoch for "{self.network_name}"', markers=True).show()

    def plot_confusion_matrix(self, test_generator):
        if self.model == None: 
            print('=== No model to test ===')
            return
        y_pred = self.model.predict(test_generator)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = test_generator.classes
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    def plot_data_distribution(self, train_labels, test_labels):
        # Calculate class counts for training and testing data
        train_class_counts = [len([x for x in train_labels if x == label]) for label in self.class_labels.keys()]
        test_class_counts =  [len([x for x in test_labels  if x == label]) for label in self.class_labels.keys()]

        fig = go.Figure()

        # Plotting training data types
        fig.add_trace(go.Pie(labels=[label.title() for label in self.class_labels.keys()], 
                            values=train_class_counts, 
                            marker=dict(colors=['#FAC500', '#0BFA00', '#0066FA', '#FA0000']), 
                            textinfo='percent+label+value', 
                            textfont=dict(size=20),
                            hole=0.3,
                            pull=[0.1, 0.1, 0.1, 0.1],
                            domain={'x': [0, 0.3], 'y': [0.5, 1]}))

        # Plotting distribution of train test split
        fig.add_trace(go.Pie(labels=['Train', 'Test'], 
                            values=[len(train_labels), len(test_labels)], 
                            marker=dict(colors=['darkcyan', 'orange']), 
                            textinfo='percent+label+value', 
                            textfont=dict(size=20),
                            hole=0.3,
                            pull=[0.1, 0],
                            domain={'x': [0.35, 0.65], 'y': [0.5, 1]}))

        # Plotting testing data types
        fig.add_trace(go.Pie(labels=[label.title() for label in self.class_labels.keys()], 
                            values=test_class_counts, 
                            marker=dict(colors=['#FAC500', '#0BFA00', '#0066FA', '#FA0000']), 
                            textinfo='percent+label+value', 
                            textfont=dict(size=20),
                            hole=0.3,
                            pull=[0.1, 0.1, 0.1, 0.1],
                            domain={'x': [0.7, 1], 'y': [0.5, 1]}))

        fig.update_layout(title='Data Distribution', grid={'rows': 1, 'columns': 3})
        fig.show()

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adamax, SGD, AdamW
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,
    BatchNormalization, GlobalAveragePooling2D
)


import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.ndimage import zoom
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
            return {idx: cls for idx, cls in enumerate(classes)}
        
        return {}

    def setup_data_generators(self,
                            train_dir: str,
                            val_dir: str,
                            test_dir: str,
                            batch_size: int = 32,
                            seed: int = 42) -> Tuple[int, int, int]:
        """
        Setup data generators for training, validation, and testing.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            test_dir: Path to test data directory
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (steps_per_epoch, validation_steps, test_steps)
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
        # Training generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **default_augmentation
        )
        
        # Validation and test generators without augmentation (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Setup generators
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',  # or 'binary' for binary classification
            seed=seed,
            shuffle=True
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',  # or 'binary' for binary classification
            shuffle=False,
            seed=seed
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',  # or 'binary' for binary classification
            shuffle=False,
            seed=seed
        )
        
        # Update class labels from generator
        self.class_labels = {v: k for k, v in self.train_generator.class_indices.items()}
        self.num_classes = len(self.class_labels)
        
        # Calculate steps for each dataset
        steps_per_epoch = ceil(self.train_generator.samples / batch_size)
        validation_steps = ceil(self.val_generator.samples / batch_size)
        test_steps = ceil(self.test_generator.samples / batch_size)
        
        return steps_per_epoch, validation_steps, test_steps

    def build_model(self) -> bool:
        """Build the specified model architecture."""
        try:
            if self.model_name == 'SimpleNet':
                self.model = Sequential([
                    InputLayer(input_shape=self.img_shape),  
                    Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
                    MaxPooling2D(pool_size=2),
                    Conv2D(filters=32, kernel_size=3, activation='relu'),
                    MaxPooling2D(pool_size=2),
                    Flatten(),
                    Dense(64, activation='relu'),
                    # Dense(self.num_classes, activation='softmax')
                    Dense(1, activation='sigmoid')
                ])
            
            elif self.model_name == 'OwnV1':
                self.model = Sequential([
                    InputLayer(input_shape=self.img_shape, name='input_layer'),
                    
                    # First Conv Block
                    Conv2D(32, (3, 3), activation='relu', name='conv1'),
                    BatchNormalization(name='bn1'),
                    MaxPooling2D((2, 2), name='pool1'),
                    
                    # Second Conv Block
                    Conv2D(64, (3, 3), activation='relu', name='conv2'),
                    BatchNormalization(name='bn2'),
                    MaxPooling2D((2, 2), name='pool2'),
                    
                    # Third Conv Block
                    Conv2D(128, (3, 3), activation='relu', name='conv3'),
                    BatchNormalization(name='bn3'),
                    MaxPooling2D((2, 2), name='pool3'),

                    # Fourth Conv Block (last conv layer)
                    Conv2D(256, (3, 3), activation='relu', name='conv4_last'),
                    BatchNormalization(name='bn4'),
                    MaxPooling2D((2, 2), name='pool4'),
                    
                    # Global Average Pooling
                    GlobalAveragePooling2D(name='global_avg_pool'),
                    
                    # Dense layers
                    Dense(256, activation='relu', name='dense1'),
                    Dropout(0.5, name='dropout1'),
                    
                    # Output Layer
                    # Dense(self.num_classes, activation='softmax', name='output')
                    Dense(1, activation='sigmoid', name='output')
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
                    # Dense(self.num_classes, activation='softmax')
                    Dense(1, activation='sigmoid')
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
                    # Dense(self.num_classes, activation='softmax')
                    Dense(1, activation='sigmoid')
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
                        # Dense(self.num_classes, activation='softmax')
                        Dense(1, activation='sigmoid')
                    ])
                else:
                    print(f"Unknown model: {self.model_name}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error building model: {e}")
            return False
        
        
    def generate_gradcam_heatmap(self, image_path: str, conv_layer_name: Optional[str] = None, alpha: float = 0.5):
        """
        Generates a Grad-CAM heatmap for a given image.

        Args:
            image_path (str): Path to the input image file.
            conv_layer_name (Optional[str]): The name of the last convolutional layer. 
                                             If None, it's detected automatically.
            alpha (float): The transparency factor for overlaying the heatmap.

        Returns:
            A superimposed image with the heatmap overlay.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # 1. Preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale

        # 2. Find the last convolutional layer if not provided
        if conv_layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, Conv2D):
                    conv_layer_name = layer.name
                    break
            if conv_layer_name is None:
                raise ValueError("Could not find a Conv2D layer in the model.")
        
        # 3. Create the Grad-CAM model
        # This is the key part that fixes the issue for Sequential models.
        # We create the model here, after self.model is already built.
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(conv_layer_name).output, self.model.layers[-1].output]
        )

        # 4. REVISED: Compute gradients with explicit tape context
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # For binary classification with sigmoid, the prediction is the score
            loss = predictions[0][0]

        # Compute gradients of the loss with respect to the conv layer's output
        grads = tape.gradient(loss, conv_outputs)

        # Add a check to catch the error gracefully
        if grads is None:
            raise ValueError(
                "Gradient is None. Check that all layers between your last conv "
                f"layer ('{conv_layer_name}') and the output are differentiable. "
                "This can also happen with some Keras/TensorFlow versions."
            )

        # 5. REVISED: Pool gradients and create heatmap
        # We use grads[0] because grads has a batch dimension we need to remove.
        pooled_grads = tf.reduce_mean(grads[0], axis=(0, 1))
        
        # Weight the feature maps by the gradients
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap) # Remove the last dimension

        # For visualization, we still normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
        heatmap = cv2.resize(heatmap, self.img_size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 6. Superimpose heatmap on the original image (same as before)
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, self.img_size)
        
        superimposed_img = cv2.addWeighted(original_img, 1.0 - alpha, heatmap, alpha, 0)
        
        return superimposed_img
    

    def train(self, 
              train_dir: str,
              val_dir: str,
              test_dir: str,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              optimizer: str = 'adam',
              early_stopping: bool = True,
              save_best: bool = True,
              checkpoint_dir: str = './checkpoints',
              verbose: int = 1) -> Dict:
        """
        Train the model.
        
        Args:
            train_dir: Path to training data
            test_dir: Path to test data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer: Optimizer type ('adam', 'adamax', 'sgd')
            early_stopping: Whether to use early stopping
            save_best: Whether to save best model
            checkpoint_dir: Directory to save checkpoints
            verbose: Verbosity level
            
        Returns:
            Dictionary with training results
        """
        start_time = time.time()
    
        # Setup data generators
        # steps_per_epoch, validation_steps = self.setup_data_generators(
        #     train_dir, test_dir, batch_size
        # )
        steps_per_epoch, validation_steps, test_steps = self.setup_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=batch_size
        )
        
        
        # Build model
        if not self.build_model():
            return {'error': 'Failed to build model'}
        
        # Compile model
        #region
        optimizers = {
            'adam': Adam(
                learning_rate=learning_rate, 
                beta_1=0.9, 
                beta_2=0.999,
                epsilon=1e-7  # Better numerical stability
            ),
            'adamw': AdamW(  # Often better for vision tasks
                learning_rate=learning_rate,
                weight_decay=0.01
            ),
            'sgd': SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True  # Often helps with convergence
            )
        }
        
        self.model.compile(
            optimizer=optimizers.get(optimizer, Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.F1Score(name='f1_score')  # Available in TF 2.15+
            ]
        )
        
        if verbose > 0:
            self.model.summary()
        #endregion
        
        # Setup callbacks
        #region
        callbacks = []
        
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_auc',  # Better than val_loss for medical imaging
                patience=10,        # Increase patience for medical data
                restore_best_weights=True,
                verbose=verbose,
                mode='max'  # AUC should be maximized
            ))
        
        callbacks.append(ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,        # Less aggressive reduction
            patience=5,        # Reduce patience
            min_lr=1e-8,       # Lower minimum
            verbose=verbose,
            mode='max'
        ))
        
        if save_best:
            os.makedirs(checkpoint_dir, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model.epoch{epoch:02d}-val_acc{val_accuracy:.4f}.hdf5'),
                monitor='val_auc',              # Better metric for medical imaging
                save_best_only=True,
                save_weights_only=False,
                verbose=verbose,
                mode='max'
            ))
        #endregion
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate model
        evaluation_results = self.model.evaluate(self.test_generator, verbose=0)
        metric_names = self.model.metrics_names  # ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1_score']
        results_dict = dict(zip(metric_names, evaluation_results))
        training_time = time.time() - start_time
        
        results = {
            'loss': results_dict['loss'],
            'accuracy': results_dict['accuracy'],
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            # optionally add the rest:
            'precision': results_dict['precision'],
            'recall': results_dict['recall'],
            'auc': results_dict['auc'],
            'f1_score': results_dict['f1_score'],
        }
        
        print("\nTraining completed!")
        print(f"Final accuracy: {results_dict['accuracy']:.4f}")
        print(f"Final loss: {results_dict['loss']:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        return results
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        raw_pred = self.model.predict(img_array, verbose=0)

        if self.num_classes == 2:
            # Binary classification: sigmoid → float output
            prob_class_1 = float(raw_pred[0])
            prob_class_0 = 1.0 - prob_class_1

            # Map labels to values
            class_names = list(self.class_labels.keys())
            
            predicted_idx = 1 if prob_class_1 >= 0.5 else 0
            
            # print(predicted_idx)
            # print(self.class_labels)
            # print(self.class_labels[predicted_idx])
            
            predicted_class = self.class_labels[predicted_idx]
            confidence = max(prob_class_0, prob_class_1)

            probabilities = {
                self.class_labels[0]: prob_class_0,
                self.class_labels[1]: prob_class_1
            }

        else:
            # Multiclass
            predictions = raw_pred[0]
            predicted_idx = np.argmax(predictions)
            idx_to_class = {v: k for k, v in self.class_labels.items()}

            predicted_class = idx_to_class[predicted_idx]
            confidence = predictions[predicted_idx]
            probabilities = {idx_to_class[i]: float(prob) for i, prob in enumerate(predictions)}

        return predicted_class, confidence, probabilities
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str):
        """Save the current model."""
        if self.model is None:
            print("No model to save")
            return
        self.model.save(model_path)
        print(f"Model saved to {model_path}")


    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, normalize: bool = False):
        """Plot confusion matrix."""
        if self.model is None or self.test_generator is None:
            print("Model or test data not available")
            return
        
        # Get predictions
        y_pred = self.model.predict(self.test_generator, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.test_generator.classes
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = [self.class_labels[i] for i in range(self.num_classes)]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}',
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        class_names = [self.class_labels[i] for i in range(self.num_classes)]
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=class_names))

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


if __name__ == "__main__":
    # Initialize classifier
    train_dir = './data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train'
    val_dir = './data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/val'
    test_dir = './data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/test'
    
    classifier = Classifier(
        model_name='OwnV1',
        img_size=(150, 150),
        data_dir=test_dir
    )
     
    # Train the model
    # results = classifier.train(
    #     train_dir=train_dir,
    #     val_dir=val_dir,
    #     test_dir=test_dir,
    #     epochs=30,
    #     batch_size=32,
    #     learning_rate=0.001
    # )
    
    # Load the model
    # classifier.load_model('./checkpoints/Bone_fracture_OwnV1_epoch_17_val_1-00000.h5')
    classifier.load_model('checkpoints/model.epoch19-val_acc0.9940.hdf5')
    
    
    # Provide the path to an image you want to inspect
    image_to_test = 'data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/val/fractured/54d38979da5d67c85b1df0919c12d4_jumbo.jpeg'
    
    # Get the class prediction
    predicted_class, confidence, _ = classifier.predict(image_to_test)
    print(f"Prediction for '{image_to_test}': {predicted_class} with {confidence:.2%} confidence.")

    # Generate the heatmap
    # You can specify the last conv layer name from your OwnV1 model ('conv4_last')
    # or let the function find it automatically by passing None.
    heatmap_img = classifier.generate_gradcam_heatmap(image_to_test, conv_layer_name='conv4_last')
    
    # Display the result using matplotlib
    plt.figure(figsize=(8, 8))
    # OpenCV loads images in BGR, matplotlib expects RGB. We need to convert.
    plt.imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM Heatmap (Predicted: {predicted_class})")
    plt.axis('off')
    plt.savefig('fig.png')
    
    # Make predictions
    # filename = './fractno.png'
    # prediction, confidence, probabilities = classifier.predict(filename)
    # print(f"File: {filename} Predicted: {prediction} (confidence: {confidence:.2f})")
    
    # filename = './fractno.png'
    # prediction, confidence, probabilities = classifier.predict(filename)
    # print(f"File: {filename} Predicted: {prediction} (confidence: {confidence:.2f})")

    # Plot results
    # classifier.plot_training_history()
    # classifier.plot_confusion_matrix()
    
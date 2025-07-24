import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from math import ceil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import cv2
from PIL import Image, ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adamax, SGD, AdamW
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,
    BatchNormalization, GlobalAveragePooling2D
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)


class Classifier:
    
    AVAILABLE_MODELS = [
        'OwnV2', 'OwnV1', 'SimpleNet', 'VGG16', 'VGG19', 'AlexNet', 
        'InceptionV3', 'EfficientNetV2', 'ResNet50', 'InceptionResNetV2'
    ]
    
    def __init__(self, 
                 model_name: str = 'OwnV1',
                 img_size: Tuple[int, int] = (150, 150),
                 data_dir: Optional[str] = None,
                 class_labels: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize the Classifier with specified configuration.
        
        Args:
            model_name: Name of the model architecture to use
            img_size: Target image dimensions as (width, height)
            data_dir: Path to the root data directory containing class subdirectories
            class_labels: Optional manual mapping of class names to indices.
                         If None, will auto-detect from directory structure.
        
        Raises:
            ValueError: If model_name is not in AVAILABLE_MODELS
        """
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Available models: {self.AVAILABLE_MODELS}")
        
        
        self.model_name = model_name
        self.img_size = img_size
        self.img_shape = (*img_size, 3)
        self.data_dir = data_dir
        self.train_dir, self.val_dir, self.test_dir = self._detect_data_structure()
        
        # Auto-detect classes from directory structure or use provided labels
        self.class_labels = class_labels or self._detect_classes()
        self.num_classes = len(self.class_labels)
        
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Configure GPU settings and optimize TensorFlow environment."""
        # Configure GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except RuntimeError:
                pass
    
        # Suppress TensorFlow info/warning messages
        # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def _detect_classes(self) -> Dict[int, str]:
        """
        Auto-detect class labels from directory structure.
        
        Expects directory structure like:
        data_dir/
        ├── train/
        │   ├── class1/
        │   └── class2/
        ├── test/
        │   ├── class1/
        │   └── class2/
        └── val/ (optional)
            ├── class1/
            └── class2/
        
        Returns:
            Dictionary mapping class indices to class names
            
        Raises:
            SystemExit: If data_dir is None or train directory doesn't exist
        """
        if not self.data_dir:
            print("Error: No data directory specified")
            exit(1)
        
        train_dir = Path(self.data_dir) / 'train'
        if not train_dir.exists():
            train_dir = Path(self.data_dir)
        
        if train_dir.exists():
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            return {idx: cls for idx, cls in enumerate(classes)}
        
        print(f"Error: Training directory not found at {train_dir}")
        exit(1)

    def _detect_data_structure(self) -> Tuple[str, Optional[str], str]:
        """
        Automatically detect train, validation, and test directories.
        
        Returns:
            Tuple of (train_dir, val_dir, test_dir) paths
        """
        if not self.data_dir:
            raise ValueError("No data directory specified")
            
        data_path = Path(self.data_dir)
        
        # Check for standard structure: data_dir/train/, data_dir/test/, data_dir/val/
        train_dir = data_path / 'train'
        test_dir = data_path / 'test'
        val_dir = data_path / 'val'
        
        if train_dir.exists() and test_dir.exists():
            val_path = str(val_dir) if val_dir.exists() else None
            return str(train_dir), val_path, str(test_dir)
        
        # If standard structure not found, raise an error
        raise ValueError(
            f"Expected directory structure not found in {self.data_dir}. "
            "Expected: train/ and test/ directories (val/ is optional)"
        )
    
    def setup_data_generators(self,
                            train_dir: Optional[str] = None,
                            val_dir: Optional[str] = None,
                            test_dir: Optional[str] = None,
                            batch_size: int = 16,
                            seed: int = 42,
                            val_split: float = 0.2) -> Tuple[int, int, int]:
        """
        Setup data generators for training, validation, and testing with augmentation.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory  
            test_dir: Path to test data directory
            batch_size: Batch size for data loading
            seed: Random seed for reproducibility
            val_split: Fraction of training data to reserve for validation (if val_dir is missing)
            
        Returns:
            Tuple containing (steps_per_epoch, validation_steps, test_steps)
        """
        # Use auto-detected directories if not provided
        train_dir = train_dir or self.train_dir
        val_dir = val_dir or self.val_dir
        test_dir = test_dir or self.test_dir
        
        # Default augmentation config
        augmentation_config = {
            'rotation_range': 20,
            'brightness_range': (0.8, 1.2),
            'width_shift_range': 0.015,
            'height_shift_range': 0.015,
            'shear_range': 20,
            'horizontal_flip': True,
        }
        
        # Create data generators
        train_datagen = ImageDataGenerator(rescale=1./255, **augmentation_config)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Determine class_mode based on number of classes
        class_mode = 'binary' if self.num_classes == 2 else 'categorical'
        
        # Setup generators
        if val_dir and os.path.exists(val_dir) and os.path.isdir(val_dir):
            # Use explicit validation directory
            train_datagen = ImageDataGenerator(rescale=1./255, **augmentation_config)
            val_datagen = ImageDataGenerator(rescale=1./255)

            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode=class_mode,
                seed=seed,
                shuffle=True
            )

            self.val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode=class_mode,
                shuffle=False,
                seed=seed
            )
        else:
            # Fallback: split training data into train+val
            train_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split, **augmentation_config)

            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode=class_mode,
                subset='training',
                seed=seed,
                shuffle=True
            )

            self.val_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode=class_mode,
                subset='validation',
                seed=seed,
                shuffle=False
            )

        # Always set up test generator the same
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False,
            seed=seed
        )

        # Update class labels
        self.class_labels = {v: k for k, v in self.train_generator.class_indices.items()}
        self.num_classes = len(self.class_labels)
        
        # Calculate steps for training
        steps_per_epoch = ceil(self.train_generator.samples / batch_size)
        validation_steps = ceil(self.val_generator.samples / batch_size)
        test_steps = ceil(self.test_generator.samples / batch_size)
        
        print("Data generators setup complete:")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        
        return steps_per_epoch, validation_steps, test_steps

    def build_model(self) -> bool:
        """
        Build the specified model architecture.
        
        Supports custom architectures (SimpleNet, OwnV1, OwnV2, AlexNet) and
        transfer learning models (VGG16, VGG19, ResNet50, InceptionV3, etc.).
        
        Returns:
            True if model was built successfully, False otherwise
        """
        try:
            if self.model_name == 'OwnV1':
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
                ])
                
                # Add appropriate output layer
                if self.num_classes == 2:
                    self.model.add(Dense(1, activation='sigmoid', name='output'))
                else:
                    self.model.add(Dense(self.num_classes, activation='softmax', name='output'))
            
            
            elif self.model_name == 'OwnV2':
                # More advanced architecture with residual-like connections
                self.model = Sequential([
                    InputLayer(input_shape=self.img_shape, name='input_layer'),
                    
                    # First Conv Block
                    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1'),
                    BatchNormalization(name='bn1'),
                    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
                    MaxPooling2D((2, 2), name='pool1'),
                    Dropout(0.25, name='dropout1'),
                    
                    # Second Conv Block
                    Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2'),
                    BatchNormalization(name='bn2'),
                    Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
                    MaxPooling2D((2, 2), name='pool2'),
                    Dropout(0.25, name='dropout2'),
                    
                    # Third Conv Block
                    Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3'),
                    BatchNormalization(name='bn3'),
                    Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
                    MaxPooling2D((2, 2), name='pool3'),
                    Dropout(0.25, name='dropout3'),

                    # Fourth Conv Block
                    Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_last'),
                    BatchNormalization(name='bn4'),
                    Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'),
                    MaxPooling2D((2, 2), name='pool4'),
                    Dropout(0.25, name='dropout4'),
                    
                    # Global Average Pooling
                    GlobalAveragePooling2D(name='global_avg_pool'),
                    
                    # Dense layers
                    Dense(512, activation='relu', name='dense1'),
                    Dropout(0.5, name='dropout_dense1'),
                    Dense(256, activation='relu', name='dense2'),
                    Dropout(0.5, name='dropout_dense2'),
                ])
                
                # Add appropriate output layer
                if self.num_classes == 2:
                    self.model.add(Dense(1, activation='sigmoid', name='output'))
                else:
                    self.model.add(Dense(self.num_classes, activation='softmax', name='output'))
            
            
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
                ])
                
                # Add appropriate output layer
                if self.num_classes == 2:
                    self.model.add(Dense(1, activation='sigmoid'))
                else:
                    self.model.add(Dense(self.num_classes, activation='softmax'))
            
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
                    ])
                    
                    # Add appropriate output layer
                    if self.num_classes == 2:
                        self.model.add(Dense(1, activation='sigmoid'))
                    else:
                        self.model.add(Dense(self.num_classes, activation='softmax'))
                else:
                    print(f"Unknown model: {self.model_name}")
                    return False
            
            print(f"Model built for {'binary' if self.num_classes == 2 else 'multi-class'} classification")
            return True
            
        except Exception as e:
            print(f"Error building model: {e}")
            return False
        
        
    def generate_gradcam_heatmap(self, 
                               image_path: str, 
                               conv_layer_name: Optional[str] = None, 
                               alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap for model interpretability.
        
        Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts
        of the image are important for the model's prediction by highlighting regions
        that strongly influence the classification decision.
        
        Args:
            image_path: Path to the input image file
            conv_layer_name: Name of the convolutional layer to analyze.
                           If None, automatically uses the last Conv2D layer
            alpha: Transparency factor for heatmap overlay (0.0-1.0)
            
        Returns:
            Tuple containing:
                - superimposed_img: Original image with heatmap overlay
                - heatmap: Raw heatmap visualization
                - original_img: Original preprocessed image
                
        Raises:
            ValueError: If model hasn't been trained/loaded or no Conv2D layers found
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # Preprocess the input image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Find the last convolutional layer if not specified
        if conv_layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, Conv2D):
                    conv_layer_name = layer.name
                    break
            if conv_layer_name is None:
                raise ValueError("Could not find a Conv2D layer in the model.")
        
        # Create Grad-CAM model
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(conv_layer_name).output, self.model.layers[-1].output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if self.num_classes == 2:
                # Binary classification with sigmoid
                # predicted_class = tf.cast(predictions[0][0] > 0.5, tf.float32)
                # loss = predicted_class * predictions[0][0] + (1 - predicted_class) * (1 - predictions[0][0])
                loss = predictions[0][0]
            else:
                # Multi-class classification with softmax
                predicted_class_idx = tf.argmax(predictions[0])
                loss = predictions[0][predicted_class_idx]
            
        # Calculate gradients of loss with respect to conv layer output
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError(
                f"Gradient is None. Check that all layers between '{conv_layer_name}' "
                "and the output are differentiable."
            )

        # Pool gradients and create heatmap
        pooled_grads = tf.reduce_mean(grads[0], axis=(0, 1))
        pooled_grads = pooled_grads / (tf.norm(pooled_grads) + 1e-8)
        
        # Weight feature maps by gradients
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize heatmap for visualization
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
        heatmap = cv2.resize(heatmap, self.img_size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose heatmap on original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, self.img_size)
        
        superimposed_img = cv2.addWeighted(original_img, 1.0 - alpha, heatmap, alpha, 0)
        
        return superimposed_img, heatmap, original_img
    
    # Hard coded layer names for OwnV1 model
    def test_multiple_layers(self, image_path: str, alpha: float = 0.5) -> None:
        """
        Test Grad-CAM visualization with multiple convolutional layers.
        
        Creates a 2x2 grid showing Grad-CAM heatmaps for different layers,
        useful for understanding how different network depths capture features.
        
        Args:
            image_path: Path to the input image
            alpha: Transparency factor for heatmap overlay
        """
        layers_to_test = ['conv1', 'conv2', 'conv3', 'conv4_last']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, layer_name in enumerate(layers_to_test):
            try:
                superimposed_img, _, _ = self.generate_gradcam_heatmap(
                    image_path, conv_layer_name=layer_name, alpha=alpha
                )
                
                axes[i].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
                axes[i].set_title(f'Layer: {layer_name}')
                axes[i].axis('off')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error with {layer_name}:\n{str(e)}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Layer: {layer_name} (Error)')
        
        plt.tight_layout()
        plt.savefig('gradcam_layers_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    

    def train(self, 
              train_dir: Optional[str] = None,
              val_dir: Optional[str] = None,
              test_dir: Optional[str] = None,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              optimizer: str = 'adam',
              early_stopping: bool = True,
              save_best: bool = True,
              checkpoint_dir: str = './checkpoints',
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the model with comprehensive monitoring and callbacks.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            test_dir: Path to test data directory
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            early_stopping: Whether to use early stopping based on validation AUC
            save_best: Whether to save the best model during training
            checkpoint_dir: Directory to save model checkpoints
            verbose: Verbosity level
            
        Returns:
            Dictionary containing training results and metrics
        """
        start_time = time.time()
    
        # Setup data generators
        if train_dir and val_dir and test_dir:
            steps_per_epoch, validation_steps, test_steps = self.setup_data_generators(
                train_dir=train_dir,
                val_dir=val_dir,
                test_dir=test_dir,
                batch_size=batch_size
            )
        else:
            steps_per_epoch, validation_steps, test_steps = self.setup_data_generators(
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
                epsilon=1e-7
            ),
            'adamw': AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01
            ),
            'sgd': SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        }
        
        # Compile model
        
        loss_function = 'binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy'
        
        self.model.compile(
            optimizer=optimizers.get(optimizer, Adam(learning_rate=learning_rate)),
            loss=loss_function,
            metrics=[
                'accuracy',
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall'),
                # tf.keras.metrics.AUC(name='auc'),
                # tf.keras.metrics.F1Score(name='f1_score')
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
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=verbose,
                mode='max'
            ))
        
        callbacks.append(ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose,
            mode='max'
        ))
        
        if save_best:
            os.makedirs(checkpoint_dir, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model.epoch{epoch:02d}-val_acc{val_accuracy:.4f}.hdf5'),
                monitor='val_accuracy',
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
            # 'precision': results_dict['precision'],
            # 'recall': results_dict['recall'],
            # 'auc': results_dict['auc'],
            # 'f1_score': results_dict['f1_score'],
            'classification_type': 'binary' if self.num_classes == 2 else 'multi-class',
            'num_classes': self.num_classes
        }
        
        if verbose > 0:
            print("\nTraining completed!")
            print(f"Final accuracy: {results['accuracy']:.4f}")
            # print(f"Final AUC: {results['auc']:.4f}")
            print(f"Training time: {training_time:.2f} seconds")
        
        return results
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to the image file to classify
            
        Returns:
            Tuple containing:
                - predicted_class: Name of the predicted class
                - confidence: Confidence score for the prediction
                - probabilities: Dictionary with probabilities for all classes
                
        Raises:
            ValueError: If model hasn't been trained or loaded
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
            # Binary classification
            prob_class_1 = float(raw_pred[0])
            prob_class_0 = 1.0 - prob_class_1
            
            predicted_idx = 1 if prob_class_1 >= 0.5 else 0
            predicted_class = self.class_labels[predicted_idx]
            confidence = max(prob_class_0, prob_class_1)

            probabilities = {
                self.class_labels[0]: prob_class_0,
                self.class_labels[1]: prob_class_1
            }

        else:
            # Multiclass classification
            predictions = raw_pred[0]
            predicted_idx = np.argmax(predictions)
            
            predicted_class = self.class_labels[predicted_idx]
            confidence = float(predictions[predicted_idx])
            probabilities = {self.class_labels[i]: float(prob) 
                           for i, prob in enumerate(predictions)}

        return predicted_class, confidence, probabilities
    
    def load_model(self, model_path: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str) -> None:
        """
        Save the current model to disk.
        
        Args:
            model_path: Path where the model should be saved
            
        Raises:
            ValueError: If no model exists to save
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")



    def plot_confusion_matrix(self, normalize: bool = False, save_path: Optional[str] = None):
        if self.model is None or self.test_generator is None:
            print("Model or test data not available")
            return

        y_true = self.test_generator.classes
        y_pred = self.model.predict(self.test_generator, verbose=0)
        
        if self.num_classes == 2:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', xticklabels=self.class_labels.values(), 
                    yticklabels=self.class_labels.values())
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = save_path or f'confusion_matrix_{int(time.time())}.png'
        plt.savefig(save_path)
        plt.close()

    def plot_training_history(self, save_path: Optional[str] = None):
        if self.history is None:
            print("No training history available")
            return

        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path = save_path or f'training_history_{int(time.time())}.png'
        plt.savefig(save_path)
        plt.close()
    
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
    # data_dir = 'data_bone_frac/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification'
    data_dir = 'data_pneumonia_final_balanced'
    # data_dir = 'data_mura/data_mura_dir'
    # data_dir = 'data_alzheimer/data_split'
    
    classifier = Classifier(
        model_name='VGG19',  # 'OwnV1', 'AlexNet', 'ResNet50', 'InceptionV3', etc.
        img_size=(150, 150),
        # img_size=(224, 224),
        data_dir=data_dir
    )
    
    # Train the model
    results = classifier.train(
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
    )
    
    # Load the model
    # classifier.load_model('checkpoints/model.epoch20-val_acc0.8750.hdf5')
    
    image_to_test = 'data_pneumonia/train/PNEUMONIA/person1657_bacteria_4399.jpeg'
    predicted_class, confidence, _ = classifier.predict(image_to_test)
    print(f"Prediction for '{image_to_test}': {predicted_class} with {confidence:.2%} confidence.")
    
    # heatmap_img = classifier.generate_gradcam_heatmap(image_to_test, conv_layer_name='conv4_last')
    classifier.test_multiple_layers(image_to_test)
    classifier.plot_training_history()
    classifier.plot_confusion_matrix()

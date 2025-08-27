import os
import time
import warnings
import logging
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
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,
    BatchNormalization, GlobalAveragePooling2D,
    Input, ReLU, LeakyReLU, Add
)

#region Env Setup
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
#endregion

class Classifier:
    
    AVAILABLE_MODELS = [
        'OwnV3', 'OwnV2', 'OwnV1', 'VGG16', 'VGG19', 'AlexNet', 
        'InceptionV3', 'EfficientNetV2', 'ResNet50', 'InceptionResNetV2'
    ]
        
    def __init__(self, 
                 model_name: str = 'OwnV1',
                 img_size: Tuple[int, int] = (150, 150),
                 data_dir: Optional[str] = None) -> None:
        """
        Initialize the Classifier with specified configuration.
        
        Args:
            model_name: Name of the model architecture to use
            img_size: Target image dimensions as (width, height)
            data_dir: Path to the root data directory containing class subdirectories\n
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
        
        Raises:
            ValueError: If model_name is not in AVAILABLE_MODELS
        """
        logger.info(f"Initializing Classifier with model: {model_name}, image size: {img_size}")
        
        if model_name not in self.AVAILABLE_MODELS:
            logger.error(f"Unsupported model '{model_name}'. Available: {self.AVAILABLE_MODELS}")
            raise ValueError(f"Unsupported model '{model_name}'. Available: {self.AVAILABLE_MODELS}")
        
        self.model_name = model_name
        self.img_size = img_size
        self.img_shape = (*img_size, 3)
        self.data_dir = data_dir
        
        self.train_dir, self.val_dir, self.test_dir = self._detect_data_structure()
        
        # Auto-detect classes from directory structure or use provided labels
        self.class_labels = self._detect_classes()
        self.num_classes = len(self.class_labels)
        
        # Validate classification setup
        if self.num_classes == 2:
            logger.info("Binary classification mode detected")
        elif self.num_classes > 10:
            logger.warning(f"Large number of classes ({self.num_classes}) detected.\nConsider if this is intentional.")
        else:
            logger.info(f"Multi-class classification mode with {self.num_classes} classes")
        
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        self._setup_environment()
        logger.info(f"Classifier initialization complete. Classes: {list(self.class_labels.values())}")

    def _setup_environment(self) -> None:
        """Configure GPU settings and optimize TensorFlow environment."""
        logger.info("Setting up TensorFlow environment...")
        
        # Configure GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logger.info(f"GPU memory growth enabled for device: {physical_devices[0]}")
            except RuntimeError as e:
                logger.warning(f"Could not set GPU memory growth: {e}")
        else:
            logger.info("No GPU devices found, using CPU")
    
        # Suppress TensorFlow info/warning messages
        # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logger.info("TensorFlow environment setup complete")

    def _detect_classes(self) -> Dict[int, str]:
        """
        Auto-detect class labels from directory structure.
        
        Returns:
            Dictionary mapping class indices to class names
            
        Raises:
            SystemExit: If data_dir is None or train directory doesn't exist
        """
        logger.info("Detecting classes from directory structure...")
        
        if not self.data_dir:
            logger.error("No data directory specified")
            exit(1)
        
        train_dir = Path(self.data_dir) / 'train'
        if not train_dir.exists():
            train_dir = Path(self.data_dir)
            logger.info(f"Standard train/ directory not found, using root: {train_dir}")
        
        if train_dir.exists():
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            logger.info(f"Found {len(classes)} valid classes: {classes}")
            return {idx: cls for idx, cls in enumerate(classes)}
        
        logger.error( f"Train directory '{train_dir}' does not exist. ")
        exit(1)

    def _detect_data_structure(self) -> Tuple[str, Optional[str], str]:
        """
        Automatically detect train, validation, and test directories.
        
        Returns:
            Tuple of (train_dir, val_dir, test_dir) paths
        """
        if not self.data_dir:
            logger.error("No data directory specified")
            raise ValueError("No data directory specified")
            
        data_path = Path(self.data_dir)
        logger.info(f"Analyzing data structure in: {data_path}")
        
        # Check for standard structure: data_dir/train/, data_dir/test/, data_dir/val/
        train_dir = data_path / 'train'
        test_dir = data_path / 'test'
        val_dir = data_path / 'val'
        
        if train_dir.exists() and test_dir.exists():
            val_path = str(val_dir) if val_dir.exists() else None
            logger.info(f"Standard structure detected - Train: {train_dir}, "
                       f"Test: {test_dir}, Val: {val_path or 'None (will split from train)'}")
            return str(train_dir), val_path, str(test_dir)
        
        # If standard structure not found, raise an error
        logger.error(f"Expected directory structure not found in {self.data_dir}")
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
        
        Returns:
            Tuple containing (steps_per_epoch, validation_steps, test_steps)
        """
        logger.info("Setting up data generators...")
        
        # Use auto-detected directories if not provided
        train_dir = train_dir or self.train_dir
        val_dir = val_dir or self.val_dir
        test_dir = test_dir or self.test_dir
        
        logger.info(f"  Data directories - Train: {train_dir}, Val: {val_dir}, Test: {test_dir}")
        logger.info(f"  Batch size: {batch_size}, Validation split: {val_split}")
        
        # Default augmentation config
        augmentation_config = {
            'rotation_range': 20,
            'brightness_range': (0.8, 1.2),
            'width_shift_range': 0.015,
            'height_shift_range': 0.015,
            'shear_range': 20,
            'horizontal_flip': True,
        }
        logger.info(f"  Using augmentation config: {augmentation_config}")
        
        # Determine class_mode based on number of classes
        class_mode = 'binary' if self.num_classes == 2 else 'categorical'
        logger.info(f"  Using class_mode: {class_mode}")
        
        # Setup generators
        if val_dir and os.path.exists(val_dir) and os.path.isdir(val_dir):
            logger.info("  Using explicit validation directory")
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
            logger.info(f"  Splitting training data into train/val with ratio {1-val_split:.1f}/{val_split:.1f}")
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

        # Always set up test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False,
            seed=seed
        )

        # Update class labels from generator
        self.class_labels = {v: k for k, v in self.train_generator.class_indices.items()}
        self.num_classes = len(self.class_labels)
        
        # Calculate steps for training
        steps_per_epoch = ceil(self.train_generator.samples / batch_size)
        validation_steps = ceil(self.val_generator.samples / batch_size)
        test_steps = ceil(self.test_generator.samples / batch_size)
        
        # Check for class imbalance
        self._check_class_imbalance()
        
        logger.info("Data generators setup complete:")
        logger.info(f"  Training samples: {self.train_generator.samples} ({steps_per_epoch} steps)")
        logger.info(f"  Validation samples: {self.val_generator.samples} ({validation_steps} steps)")
        logger.info(f"  Test samples: {self.test_generator.samples} ({test_steps} steps)")
        logger.info(f"  Class mapping: {self.class_labels}")
        
        return steps_per_epoch, validation_steps, test_steps

    def _check_class_imbalance(self) -> None:
        """Check for class imbalance and log warnings if found."""
        if not self.train_generator:
            return
        
        # Get class distribution
        class_counts = {}
        for class_idx, class_name in self.class_labels.items():
            count = sum(1 for label in self.train_generator.classes if label == class_idx)
            class_counts[class_name] = count
        
        logger.info(f"Training class distribution: {class_counts}")
        
        # Check for imbalance
        counts = list(class_counts.values())
        max_count, min_count = max(counts), min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            logger.warning(f"Significant class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
            logger.warning("Consider using class weights or data balancing techniques")
        else:
            logger.info(f"Class balance looks good. Ratio: {imbalance_ratio:.2f}:1")


    def build_model(self) -> bool:
        """
        Build the specified model architecture.
        """
        logger.info(f"Building model: {self.model_name}")
        logger.info(f"Input shape: {self.img_shape}, Output classes: {self.num_classes}")
        
        try:
            if self.model_name == 'OwnV1':
                logger.info("Building custom OwnV1 architecture")
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
                    logger.info("Added sigmoid output layer for binary classification")
                else:
                    self.model.add(Dense(self.num_classes, activation='softmax', name='output'))
                    logger.info(f"Added softmax output layer for {self.num_classes}-class classification")
            
            elif self.model_name == 'OwnV2':
                logger.info("Building custom OwnV2 architecture")
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

                    # Fourth Conv Block | Last
                    Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4'),
                    BatchNormalization(name='bn4'),
                    Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_last'),
                    MaxPooling2D((2, 2), name='pool4'),
                    Dropout(0.25, name='dropout4'),
                    
                    # Global Average Pooling
                    GlobalAveragePooling2D(name='global_avg_pool'),
                    
                    # Dense layers
                    Dense(512, activation='relu', name='dense1'),
                    Dropout(0.3, name='dropout_dense1'),
                    Dense(256, activation='relu', name='dense2'),
                    Dropout(0.3, name='dropout_dense2'),
                ])
                
                # Add appropriate output layer
                if self.num_classes == 2:
                    self.model.add(Dense(1, activation='sigmoid', name='output'))
                    logger.info("Added sigmoid output layer for binary classification")
                else:
                    self.model.add(Dense(self.num_classes, activation='softmax', name='output'))
                    logger.info(f"Added softmax output layer for {self.num_classes}-class classification")
            
            elif self.model_name == 'OwnV3':
                logger.info("Building custom OwnV3 architecture")
                
                def residual_block(x, filters, kernel_size=(3, 3), stride=1):
                    # Main path
                    y = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer='l2')(x)
                    y = BatchNormalization()(y)
                    y = LeakyReLU(alpha=0.01)(y)

                    y = Conv2D(filters, kernel_size, padding='same', kernel_regularizer='l2')(y)
                    y = BatchNormalization()(y)

                    # Shortcut connection
                    # If strides > 1 or filter count changes, we need to project the shortcut.
                    if stride != 1 or x.shape[-1] != filters:
                        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(x)
                    else:
                        shortcut = x

                    # Add shortcut to the main path
                    out = Add()([y, shortcut])
                    out = LeakyReLU(alpha=0.01)(out)
                    return out

                # --- Input & Stem ---
                inputs = Input(shape=self.img_shape)
                x = Conv2D(32, (7, 7), strides=2, padding='same')(inputs)
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.01)(x)
                x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

                # --- Residual Blocks ---
                x = residual_block(x, filters=32)
                x = residual_block(x, filters=64, stride=2) 
                x = residual_block(x, filters=128, stride=2)
                x = residual_block(x, filters=256, stride=2)

                # --- Classifier Head ---
                x = GlobalAveragePooling2D()(x)
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.6)(x)

                # Output
                if self.num_classes == 2:
                    outputs = Dense(1, activation='sigmoid', name='output')(x)
                    logger.info("Added sigmoid output layer for binary classification")
                else:
                    outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
                    logger.info(f"Added softmax output layer for {self.num_classes}-class classification")

                self.model = Model(inputs=inputs, outputs=outputs, name='OwnV3')            
            
            elif self.model_name == 'AlexNet':
                logger.info("Building AlexNet architecture")
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
                    self.model.add(Dense(1, activation='sigmoid', name='output'))
                    logger.info("Added sigmoid output layer for binary classification")
                else:
                    self.model.add(Dense(self.num_classes, activation='softmax', name='output'))
                    logger.info(f"Added softmax output layer for {self.num_classes}-class classification")
            
            # Transfer learning models
            else:
                logger.info(f"Building transfer learning model: {self.model_name}")
                base_models = {
                    'VGG16': tf.keras.applications.VGG16,
                    'VGG19': tf.keras.applications.VGG19,
                    'ResNet50': tf.keras.applications.ResNet50,
                    'InceptionV3': tf.keras.applications.InceptionV3,
                    'EfficientNetV2': tf.keras.applications.EfficientNetV2B0,
                    'InceptionResNetV2': tf.keras.applications.InceptionResNetV2
                }
                
                if self.model_name in base_models:
                    logger.info("Loading pre-trained weights from ImageNet")
                    base_model = base_models[self.model_name](
                        weights='imagenet',
                        include_top=False,
                        input_shape=self.img_shape
                    )
                    base_model.trainable = False
                    logger.info(f"Froze base model with {len(base_model.layers)} layers")
                    
                    self.model = Sequential([
                        base_model,
                        GlobalAveragePooling2D(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                    ])
                    
                    # Add appropriate output layer
                    if self.num_classes == 2:
                        self.model.add(Dense(1, activation='sigmoid'))
                        logger.info("Added sigmoid output layer for binary classification")
                    else:
                        self.model.add(Dense(self.num_classes, activation='softmax'))
                        logger.info(f"Added softmax output layer for {self.num_classes}-class classification")
                else:
                    logger.error(f"Unknown model: {self.model_name}")
                    return False
            
            # Count parameters
            total_params = self.model.count_params()
            trainable_params = sum([K.count_params(w) for w in self.model.trainable_weights])
            logger.info("Model built successfully:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building model: {e}", exc_info=True)
            return False


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
        
        logger.info("Starting model training...")
        logger.info("Training parameters:")
        logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}, Optimizer: {optimizer}")
        logger.info(f"  Early stopping: {early_stopping}, Save best: {save_best}")
        
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
        logger.info("Building model architecture...")
        if not self.build_model():
            logger.error("Failed to build model")
            return {'error': 'Failed to build model'}
        
        # Compile model
        #region
        logger.info("Compiling model...")
        total_steps = steps_per_epoch * epochs 
        cosine_decay_schedule = CosineDecay(
            initial_learning_rate=learning_rate, # Start with a higher LR, e.g., 0.01
            decay_steps=total_steps
        )
        optimizers = {
            'adam': Adam(
                learning_rate=learning_rate, 
                beta_1=0.9, 
                beta_2=0.999,
                epsilon=1e-7
            ),
            'adamax': Adamax(
                learning_rate=learning_rate
            ),
            'adamw': AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01
            ),
            'sgd': SGD(
                learning_rate=cosine_decay_schedule,
                momentum=0.9,
                nesterov=True
            ),
        }
        
        if self.num_classes == 2:
            loss_function = 'binary_crossentropy'
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
            ]
        else:
            loss_function = 'categorical_crossentropy'
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision', average='weighted'),
                tf.keras.metrics.Recall(name='recall', average='weighted'),
                tf.keras.metrics.F1Score(name='f1_score', average='weighted')
            ]
        
        logger.info(f"Using loss function: {loss_function}")
        logger.info(f"Using metrics: {metrics}")
        
        self.model.compile(
            optimizer=optimizers.get(optimizer, Adam(learning_rate=learning_rate)),
            loss=loss_function,
            metrics=metrics
        )
        
        if verbose > 0:
            self.model.summary()
        #endregion
        
        # Setup callbacks
        #region
        logger.info("Setting up training callbacks...")
        callbacks = []
        
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_accuracy',
                patience=12,
                restore_best_weights=True,
                verbose=verbose,
                mode='max'
            ))
            logger.info("Added EarlyStopping callback (patience=10, monitor=val_accuracy)")

        
        callbacks.append(ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose,
            mode='max'
        ))
        logger.info("Added ReduceLROnPlateau callback (factor=0.5, patience=5)")

        
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
            logger.info(f"Added ModelCheckpoint callback (save to: {checkpoint_dir})")

        #endregion
        
        # Train model
        logger.info("Starting training loop...")
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        epochs_trained = len(self.history.history['loss'])
        logger.info(f"Training completed in {training_time:.2f} seconds ({epochs_trained} epochs)")
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        evaluation_results = self.model.evaluate(self.test_generator, verbose=0)
        metric_names = self.model.metrics_names
        results_dict = dict(zip(metric_names, evaluation_results))
        
        results = {
            'loss': results_dict['loss'],
            'accuracy': results_dict['accuracy'],
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            'classification_type': 'binary' if self.num_classes == 2 else 'multi-class',
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'img_size': self.img_size,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'learning_rate': learning_rate
        }
        
        # Add available metrics
        for metric in ['precision', 'recall', 'auc', 'f1_score']:
            if metric in results_dict:
                results[metric] = results_dict[metric]
        
        logger.info("Training results:")
        logger.info(f"  Final test accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Final test loss: {results['loss']:.4f}")
        logger.info(f"  Training time: {training_time:.2f} seconds")
        logger.info(f"  Epochs trained: {epochs_trained}")
        
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
        logger.info(f"Making prediction for image: {image_path}")
        
        if self.model is None:
            logger.error("Model not trained or loaded")
            raise ValueError("Model not trained or loaded")

        # Validate image file
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess image
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            logger.info(f"Image preprocessed to shape: {img_array.shape}")
        except Exception as e:
            logger.error(f"Error loading/preprocessing image: {e}")
            raise

        # Make prediction
        raw_pred = self.model.predict(img_array, verbose=0)
        logger.info(f"Raw prediction output: {raw_pred}")

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
        
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.2%}")
        return predicted_class, confidence, probabilities
    
    def evaluate(self):
        # Evaluate model
        
        if self.val_generator is None:
            logger.warning("val_generator is None — preparing data...")
            self.setup_data_generators()
        
        logger.info("Evaluating model on test set...")
        evaluation_results = self.model.evaluate(self.val_generator, verbose=0)
        metric_names = self.model.metrics_names
        results_dict = dict(zip(metric_names, evaluation_results))
        
        results = {
            'loss': results_dict['loss'],
            'accuracy': results_dict['accuracy'],
            'classification_type': 'binary' if self.num_classes == 2 else 'multi-class',
            'num_classes': self.num_classes,
            'model_name': self.model_name,
        }
        
        # Add available metrics
        for metric in ['precision', 'recall', 'auc', 'f1_score']:
            if metric in results_dict:
                results[metric] = results_dict[metric]
        
        logger.info("Training results:")
        logger.info(f"  Final test accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Final test loss: {results['loss']:.4f}")
        
        return results
    
    def evaluate_with_confidences(self):
        """
        Runs the model on self.val_generator and returns per-image predictions,
        confidences, and ground truth labels.
        """
        if self.val_generator is None:
            logger.warning("val_generator is None — preparing data...")
            self.setup_data_generators()

        # Predict probabilities for all images
        probs = self.model.predict(self.val_generator, verbose=0)

        # Binary classification → convert to shape (n_samples, 2)
        if probs.shape[1] == 1:
            probs = np.hstack([1 - probs, probs])

        preds = np.argmax(probs, axis=1)              # predicted class index
        confs = np.max(probs, axis=1)                 # confidence score
        labels = self.val_generator.classes           # true labels
        class_labels = self.class_labels              # label names

        return {
            "probs": probs,                           # (n_samples, n_classes)
            "pred_indices": preds,                    # (n_samples,)
            "confidences": confs,                      # (n_samples,)
            "true_indices": labels,                    # (n_samples,)
            "class_labels": class_labels
        }
    
    
    def load_model(self, model_path: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    
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
        logger.info(f"Model saved to {model_path}")


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
    
    def test_multiple_layers(self, image_path: str, alpha: float = 0.5) -> None: # Hard coded layer names for OwnV1 model
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
        plt.savefig('./results/gradcam_layers_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    

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
        
        save_path = save_path or f'./results/confusion_matrix_{int(time.time())}.png'
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
        save_path = save_path or f'./results/training_history_{int(time.time())}.png'
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


def evaluate_ensemble(models_info, data_dir, img_size=(150, 150), weights=None):
    """
    models_info: dict {model_name: checkpoint_path}
    Returns: ensemble accuracy
    """
    all_probs = []
    true_labels = None

    for name, ckpt in models_info.items():
        clf = Classifier(model_name=name, img_size=img_size, data_dir=data_dir)
        clf.load_model(ckpt)
        res = clf.evaluate_with_confidences()

        all_probs.append(res["probs"])
        if true_labels is None:
            true_labels = res["true_indices"]

    if weights is None:
        # Default to equal weights if none are provided
        weights = np.ones(len(all_probs)) / len(all_probs)
    
    # Calculate the weighted average of the probabilities
    avg_probs = np.average(np.stack(all_probs, axis=0), axis=0, weights=weights)
    final_preds = np.argmax(avg_probs, axis=1)

    accuracy = np.mean(final_preds == true_labels)
    logger.info(f"Ensemble accuracy: {accuracy:.4f}")

    return accuracy

def ensemble_testing():
    data_dir = 'data_pneumonia_final_balanced_og'
    
    checkpoint_paths = {
        "OwnV3": "checkpoints/Saved/OwnV3.epoch50-val_acc0.9830.hdf5",
        # "OwnV1": "checkpoints/Saved/OwnV1.epoch26-val_acc0.9761.hdf5",
        # "VGG16": "checkpoints/Saved/VGG16.epoch18-val_acc0.9534.hdf5"
        # "AlexNet": "checkpoints/Saved/AlexNet.epoch27-val_acc0.9761.hdf5",
        # "OwnV2": "checkpoints/Saved/OwnV2.epoch28-val_acc0.9705.hdf5",
        "InceptionV3": "checkpoints/Saved/InceptionV3.epoch13-val_acc0.9545.hdf5",
    }
    

    # ensemble_predict_from_checkpoints(checkpoint_paths)
    weights = [
        0.9830, 
        # 0.9761,
        # 0.9534,
        # 0.9761,
        # 0.9705,
        0.9545,
    ] 
    acc = evaluate_ensemble(checkpoint_paths, data_dir, (150, 150), weights=weights)

def train_testing():
    data_dir = 'data_pneumonia_final_balanced_og'
    #     'OwnV3', 'OwnV2', 'OwnV1', 'VGG16', 'VGG19', 'AlexNet', 
    #     'InceptionV3', 'EfficientNetV2', 'ResNet50', 'InceptionResNetV2'
    
    classifier = Classifier(
        model_name='OwnV3',
        img_size=(150, 150),
        # img_size=(224, 224),
        data_dir=data_dir
    )
    
    
    
    # Train the model
    results = classifier.train(
        epochs=100,
        batch_size=16,
        learning_rate=0.0003,
    )
    
    # Load the model
    classifier.load_model('checkpoints/Saved/OwnV3.epoch50-val_acc0.9830.hdf5')
    classifier.evaluate()
    
    
    
    image_to_test = 'data_pneumonia/train/PNEUMONIA/person1657_bacteria_4399.jpeg'
    predicted_class, confidence, _ = classifier.predict(image_to_test)
    
    heatmap_img = classifier.generate_gradcam_heatmap(image_to_test, conv_layer_name='conv4_last')
    classifier.test_multiple_layers(image_to_test)
    classifier.plot_training_history()
    classifier.plot_confusion_matrix()


if __name__ == "__main__":
    # ensemble_testing()
    # train_testing()
    ...
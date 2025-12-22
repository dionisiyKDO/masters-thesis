import os
import cv2
import time
import json
import logging
import warnings

import numpy             as np
import tensorflow        as tf
import seaborn           as sns
import matplotlib.pyplot as plt

from math       import ceil
from pathlib    import Path
from datetime   import datetime
from typing     import Dict, Optional, Tuple, Any
from PIL        import ImageFile

from sklearn.metrics            import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image   import ImageDataGenerator
from tensorflow.keras.callbacks             import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers            import Adam, SGD, AdamW
from tensorflow.keras.optimizers.schedules  import CosineDecay
from tensorflow.keras.models                import Sequential, Model
from tensorflow.keras.layers                import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,
    BatchNormalization, GlobalAveragePooling2D,
    Input, LeakyReLU, Add
)

try:
    from .progress_state import update_progress 
except ImportError:
    from progress_state  import update_progress

#region Setup env
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets" / "data_pneumonia"
    # data/ - first final version
    # data_pneumonia/ - original dataset
    # dataset_processed/ - augmented 50/50 balanced mixed
    # data_test/ - resplitted original

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
GRADCAM_DIR     = OUTPUT_DIR / "gradcam"
RESULTS_DIR     = OUTPUT_DIR / "results"

IMAGE_SIZE = (150, 150)

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#endregion

class Classifier:
    
    AVAILABLE_MODELS = [
        'OwnV4', 'OwnV3', 'OwnV2', 'OwnV1', 
        'AlexNet', 'VGG16', 'VGG19', 'ResNet50', 'DenseNet121', 'MobileNetV2',
    ]
    
    #region Initialize classifier with configuration
    
    def __init__(self, 
                 model_name: str = 'OwnV3',
                 img_size: Tuple[int, int] = IMAGE_SIZE,
                 data_dir: Path = DATA_DIR) -> None:
        """
        Initialize the Classifier with specified configuration.

        :param model_name: Name of the model architecture to use. Must be in Classifier.AVAILABLE_MODELS.
        :type model_name: str
        :param img_size: Target image dimensions as (width, height).
        :type img_size: Tuple[int, int]
        :param data_dir: Path to the root data directory. Must contain 'train/' and 'test/' subdirectories.
        :type data_dir: str
        """
        if model_name not in self.AVAILABLE_MODELS:
            logger.error(f"[INIT] Unsupported model '{self.model_name}'. Available: {self.AVAILABLE_MODELS}")
            raise ValueError(f"Unsupported model '{self.model_name}'. Available: {self.AVAILABLE_MODELS}")
            
        logger.info(f"[INIT] Initializing Classifier with model: {model_name}, size: {img_size}")
        
        self.model_name = model_name
        self.img_size = img_size
        self.img_shape = (*img_size, 3)
        self.data_dir = data_dir
        
        
        self.train_dir, self.val_dir, self.test_dir = self._detect_folders()
        self.class_labels, self.num_classes, self.class_mode = self._detect_classes()
        
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.best_checkpoint_path = None
        
        self._setup_environment()
    
    def _detect_folders(self) -> Tuple[str, Optional[str], str]:
        """
        Automatically detect train, validation, and test directories.
        
        :returns: (train_dir, val_dir, test_dir) paths.
        :rtype: Tuple[str, Optional[str], str]
        """
        if not self.data_dir:
            logger.error("[DATA] No data directory specified.")
            raise ValueError("No data directory specified")
            
        data_path = Path(self.data_dir)
        
        train_dir = data_path / 'train'
        test_dir = data_path / 'test'
        val_dir = data_path / 'val'
        
        if train_dir.exists() and test_dir.exists():
            val_dir = val_dir if val_dir.exists() else None
            logger.info("[DATA] Detected folders:")
            logger.info(f"  Train: {train_dir}")
            logger.info(f"  Test: {test_dir}")
            logger.info(f"  Val: {val_dir or 'None (will split from train)'}")
            return str(train_dir), str(val_dir), str(test_dir)
        
        logger.error(f"[DATA] Expected directory structure not found in {self.data_dir}")
        raise ValueError(
            f"Expected directory structure not found in {self.data_dir}. "
            "Expected: train/ and test/ directories (val/ is optional)"
        )
    
    def _detect_classes(self) -> Tuple[Dict[int, str], int, str]:
        """
        Auto-detect class labels from the data directory structure.
        
        :returns: (Class index mapping, number of classes, classification mode).
        :rtype: Tuple[Dict[int, str], int, str]
        """
        
        classes = sorted([d.name for d in Path(self.train_dir).iterdir() if d.is_dir()])
        num_classes = len(classes)
        class_mode = 'binary' if num_classes == 2 else 'categorical'
        
        logger.info(f"[DATA] Found {num_classes} valid classes: {classes}")
        if num_classes == 2:
            logger.info("  Binary classification mode detected.")
        elif num_classes > 10:
            logger.warning(f"[DATA] Large number of classes ({num_classes}) detected. Consider if this is intentional.")
        else:
            logger.info(f"  Multi-class classification mode with {num_classes} classes.")
                    
        return {idx: cls for idx, cls in enumerate(classes)}, num_classes, class_mode
    
    def _setup_environment(self) -> None:
        """Configure GPU settings and optimize TensorFlow environment."""
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                device_name = physical_devices[0].name if hasattr(physical_devices[0], 'name') else physical_devices[0]
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logger.info(f"[ENV] GPU memory growth enabled for device: {device_name}")
            except RuntimeError as e:
                logger.warning(f"[ENV] Could not set GPU memory growth: {e}")
        else:
            logger.info("[ENV] No GPU devices found, using CPU.")
    
        # Suppress TensorFlow info/warning messages
        # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logger.info("[ENV] Suppressing TensorFlow internal INFO/WARNING messages (level 2).")
        return
    
    #endregion
    
    
    #region Setup data pipeline
    
    def _setup_data_generators(self, batch_size: int = 16) -> Tuple[int, int, int]:
        """
        Setup data generators for training, validation, and testing with augmentation.

        :param batch_size: Number of samples per batch.
        :type batch_size: int
        :returns: (steps_per_epoch, validation_steps, test_steps).
        :rtype: Tuple[int, int, int]
        """
        logger.info(f"[DATA] Setting up data generators with batch size: {batch_size}")
        
        # Default augmentation config
        augmentation_config = {
            'rotation_range': 20,
            'brightness_range': (0.8, 1.2),
            'width_shift_range': 0.015,
            'height_shift_range': 0.015,
            'shear_range': 20,
            'horizontal_flip': True,
        }
        
        # Helper to create a generator
        def make_generator(datagen, directory, subset=None, shuffle=False):
            return datagen.flow_from_directory(
                directory,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode=self.class_mode,
                subset=subset,
                seed=42,
                shuffle=shuffle,
            )

        # Train/val setup
        if self.val_dir and os.path.isdir(self.val_dir):
            logger.info("  Using explicit validation directory.")
            train_datagen = ImageDataGenerator(rescale=1./255, **augmentation_config)
            val_datagen = ImageDataGenerator(rescale=1./255)
            self.train_generator = make_generator(train_datagen, self.train_dir, shuffle=True)
            self.val_generator = make_generator(val_datagen, self.val_dir)
        else:
            val_split = 0.2
            logger.info(f"  Splitting training data into train/val (ratio {1-val_split:.1f}/{val_split:.1f}).")
            train_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split, **augmentation_config)
            self.train_generator = make_generator(train_datagen, self.train_dir, subset="training", shuffle=True)
            self.val_generator = make_generator(train_datagen, self.train_dir, subset="validation")

        # Always set up test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = make_generator(test_datagen, self.test_dir)
        
        # Compare class indices
        generated_class_labels = {v: k for k, v in self.train_generator.class_indices.items()}
        if generated_class_labels != self.class_labels:
            logger.error("[DATA] Class labels mismatch between detected and generator.")
            raise ValueError(f"Class labels from generator do not match detected labels: {generated_class_labels} vs {self.class_labels}")
        
        # Calculate steps for training
        steps_per_epoch = ceil(self.train_generator.samples / batch_size)
        validation_steps = ceil(self.val_generator.samples / batch_size)
        test_steps = ceil(self.test_generator.samples / batch_size)
        
        # Check for class imbalance
        self._check_class_imbalance()
        
        logger.info("[DATA] Data generators setup complete:")
        logger.info(f"  Training samples: {self.train_generator.samples} ({steps_per_epoch} steps)")
        logger.info(f"  Validation samples: {self.val_generator.samples} ({validation_steps} steps)")
        logger.info(f"  Test samples: {self.test_generator.samples} ({test_steps} steps)")
        
        return steps_per_epoch, validation_steps, test_steps

    def _check_class_imbalance(self) -> None:
        """Check for class imbalance and log warnings if found."""
        # Get class distribution
        class_counts = {}
        counts = np.bincount(self.train_generator.classes)
        for class_idx, class_name in self.class_labels.items():
            class_counts[class_name] = counts[class_idx]
        
        logger.info(f"[DATA] Training class distribution: {class_counts}")
        
        # Check for imbalance
        counts = list(class_counts.values())
        max_count, min_count = max(counts), min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            logger.warning(f"[DATA] Significant class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
        else:
            logger.info(f"[DATA] Class balance ratio: {imbalance_ratio:.2f}:1")

    #endregion
    
    
    #region Get CNN model

    def load_model(self, checkpoint_path: str) -> None:
        """Load a saved model from disk."""
        self.model = tf.keras.models.load_model(checkpoint_path)
        logger.info(f"[MODEL] Model loaded from {checkpoint_path}")
    
    def _build_model(self) -> bool:
        """
        Build the specified model architecture.

        :returns: True if model built successfully, False otherwise.
        :rtype: bool
        """
        try:
            def add_output_layer(model):
                """Attach final classification layer based on num_classes."""
                if self.num_classes == 2:
                    model.add(Dense(1, activation='sigmoid', name='output'))
                    logger.info("  Added sigmoid output layer for binary classification")
                else:
                    model.add(Dense(self.num_classes, activation='softmax', name='output'))
                    logger.info(f"  Added softmax output layer for {self.num_classes}-class classification")
                return model
            
            def build_ownv1():
                model = Sequential([
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
                return add_output_layer(model)

            def build_ownv2():
                model = Sequential([
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
                return add_output_layer(model)

            def build_ownv3():
                def residual_block(x, filters, kernel_size=(3, 3), stride=1):
                    # Main path
                    y = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer='l2')(x)
                    y = BatchNormalization()(y)
                    y = LeakyReLU(alpha=0.01)(y)
                    y = Conv2D(filters, kernel_size, padding='same', kernel_regularizer='l2')(y)
                    y = BatchNormalization()(y)

                    # Shortcut connection
                    if stride != 1 or x.shape[-1] != filters:
                        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(x)
                    else:
                        shortcut = x

                    # Add shortcut to the main path
                    out = Add()([y, shortcut])
                    out = LeakyReLU(alpha=0.01)(out)
                    return out

                # Input Stem
                inputs = Input(shape=self.img_shape)
                x = Conv2D(32, (7, 7), strides=2, padding='same')(inputs)
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.01)(x)
                x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
                # Residual Blocks
                x = residual_block(x, filters=32)
                x = residual_block(x, filters=64, stride=2) 
                x = residual_block(x, filters=128, stride=2)
                x = residual_block(x, filters=256, stride=2)
                # Classifier Head
                x = GlobalAveragePooling2D()(x)
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.6)(x)

                outputs = Dense(1, activation='sigmoid', name='output')(x) 
                return Model(inputs, outputs, name='OwnV3')

            def build_ownv4():
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
                    out = Dropout(0.2)(out) 
                    return out

                # --- Input & Stem ---
                inputs = Input(shape=self.img_shape)
                x = Conv2D(32, (7, 7), strides=2, padding='same')(inputs)
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.01)(x)
                x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
                x = Dropout(0.2)(x) 

                # --- Residual Blocks ---
                x = residual_block(x, filters=32)
                x = residual_block(x, filters=64, stride=2) 
                x = residual_block(x, filters=128, stride=2)
                x = residual_block(x, filters=256, stride=2)

                # --- Classifier Head ---
                x = GlobalAveragePooling2D()(x)
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.4)(x)

                outputs = Dense(1, activation='sigmoid', name='output')(x) if self.num_classes == 2 \
                    else Dense(self.num_classes, activation='softmax', name='output')(x)

                return Model(inputs, outputs, name='OwnV4')

            def build_alexnet():
                model = Sequential([
                    Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=self.img_shape),
                    MaxPooling2D((3, 3), strides=(2, 2)),
                    BatchNormalization(),
                    Conv2D(256, (5, 5), activation='relu', padding='same'),
                    MaxPooling2D((3, 3), strides=(2, 2)),
                    BatchNormalization(),
                    Conv2D(384, (3, 3), activation='relu', padding='same'),
                    Conv2D(384, (3, 3), activation='relu', padding='same'),
                    Conv2D(256, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((3, 3), strides=(2, 2)),
                    BatchNormalization(),
                    Flatten(),
                    Dense(4096, activation='relu'),
                    Dropout(0.5),
                    Dense(4096, activation='relu'),
                    Dropout(0.5),
                ])
                return add_output_layer(model)

            def build_transfer(model_name):
                base_models = {
                    'VGG16': tf.keras.applications.VGG16,
                    'VGG19': tf.keras.applications.VGG19,
                    'ResNet50': tf.keras.applications.ResNet50,
                    'DenseNet121': tf.keras.applications.DenseNet121,
                    'MobileNetV2': tf.keras.applications.MobileNetV2,
                    # 'InceptionV3': tf.keras.applications.InceptionV3,
                    # 'EfficientNetV2': tf.keras.applications.EfficientNetV2B0,
                    # 'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
                }
                base = base_models[model_name](weights="imagenet", include_top=False, input_shape=self.img_shape)
                base.trainable = False
                model = Sequential([
                    base,
                    GlobalAveragePooling2D(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                ])
                return add_output_layer(model)

            # Map model name - builder
            builders = {
                "OwnV1": build_ownv1,
                "OwnV2": build_ownv2,
                "OwnV3": build_ownv3,
                "OwnV4": build_ownv4,
                "AlexNet": build_alexnet,
            }

            # Build chosen model
            if self.model_name in builders:
                logger.info(f"[MODEL] Building {self.model_name} architecture.")
                self.model = builders[self.model_name]()
            else:
                logger.info(f"[MODEL] Building transfer learning model: {self.model_name}")
                self.model = build_transfer(self.model_name)

            # Count parameters
            total_params = self.model.count_params()
            trainable_params = sum([K.count_params(w) for w in self.model.trainable_weights])
            
            logger.info("[MODEL] Model built successfully:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"[MODEL] Error building model: {e}")
            return False

    #endregion


    #region CNN training/predict

    def train(self, epochs=50, batch_size=32, learning_rate=0.001, optimizer='adam',
              early_stopping=True, save_best=True, checkpoint_dir=CHECKPOINT_DIR, verbose=1) -> Dict[str, Any]:
        """
        Train the model.

        :param epochs: Number of epochs to train for.
        :type epochs: int
        :param batch_size: Samples per batch.
        :type batch_size: int
        :param learning_rate: Optimizer learning rate.
        :type learning_rate: float
        :param optimizer: Name of the optimizer ('adam', 'sgd', etc.).
        :type optimizer: str
        :param early_stopping: Enable early stopping callback.
        :type early_stopping: bool
        :param save_best: Enable model checkpointing for the best weights.
        :type save_best: bool
        :param checkpoint_dir: Directory to save model checkpoints.
        :type checkpoint_dir: str
        :param verbose: Verbosity level for training (0=silent, 1=progress bar, 2=one line per epoch).
        :type verbose: int
        :returns: Dictionary of final evaluation results on the test set.
        :rtype: Dict[str, Any]
        """

        start_time = time.time()
        logger.info(f"[TRAIN] Starting model training: {self.model_name}")
        logger.info(f"  Config: Epochs={epochs}, Batch={batch_size}, Optimizer={optimizer}, LR={learning_rate}")

        def get_loss_and_metrics():
            """Return loss and metrics based on classification type."""
            if self.num_classes == 2:
                return 'binary_crossentropy', [
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                ]
            return 'categorical_crossentropy', [
                'accuracy',
                tf.keras.metrics.Precision(name='precision', average='weighted'),
                tf.keras.metrics.Recall(name='recall', average='weighted'),
                tf.keras.metrics.F1Score(name='f1_score', average='weighted'),
            ]

        def get_class_weights():
            """Balanced class weights for imbalanced datasets."""
            labels = np.array(self.train_generator.classes)
            weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            return dict(enumerate(weights))

        def get_callbacks():
            """Create training callbacks list."""
            callbacks = []
            callbacks.append(ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                            patience=5, min_lr=1e-8, verbose=verbose, mode='max'))
            if early_stopping:
                patience = 12 if epochs >= 30 else max(3, epochs // 3)
                callbacks.append(EarlyStopping(monitor='val_auc', patience=patience,
                                            restore_best_weights=True, verbose=verbose, mode='max'))
            if save_best:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                unique_checkpoint_dir = os.path.join(checkpoint_dir, f"{self.model_name}_{timestamp}")
                os.makedirs(unique_checkpoint_dir, exist_ok=True)
                static_filepath = os.path.join(unique_checkpoint_dir, "best_model.hdf5")
                
                self.best_checkpoint_path = None
        
                callbacks.append(PathTrackingModelCheckpoint(
                    classifier_instance=self, # Pass 'self' (the classifier instance)
                    filepath=static_filepath,
                    monitor='val_accuracy', save_best_only=True, save_weights_only=False, verbose=verbose, mode='max'))
            return callbacks
        
        def compile_model():
            """Compile model with chosen optimizer, loss, and metrics."""
            total_steps = steps_per_epoch * epochs
            optimizers = {
                'adam': Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                'adamw': AdamW(learning_rate=learning_rate, weight_decay=0.01),
                'sgd': SGD(learning_rate=CosineDecay(learning_rate, total_steps), momentum=0.9, nesterov=True),
            }
            loss_fn, metrics = get_loss_and_metrics()
            self.model.compile(optimizer=optimizers.get(optimizer, Adam(learning_rate=learning_rate)),
                            loss=loss_fn, metrics=metrics)
            if verbose > 0:
                self.model.summary()
            logger.info(f"[TRAIN] Model compiled with loss={loss_fn}, optimizer={optimizer}")
        
        def evaluate_training():
            """Evaluate model after training and return summary results."""
            elapsed = time.time() - start_time
            results_dict = dict(zip(self.model.metrics_names,
                                    self.model.evaluate(self.test_generator, verbose=0)))
            results = {
                'loss': results_dict['loss'],
                'accuracy': results_dict['accuracy'],
                'training_time': elapsed,
                'epochs_trained': len(self.history.history['loss']),
                'classification_type': self.class_mode,
                'num_classes': self.num_classes,
                'model_name': self.model_name,
                'img_size': self.img_size,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'learning_rate': learning_rate
            }
            for metric in ['precision', 'recall', 'auc', 'f1_score']:
                if metric in results_dict:
                    results[metric] = results_dict[metric]
            return results

        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            """
            Custom callback that reports training progress to the shared progress_state.
            Called automatically by Keras after each epoch.
            """
            def __init__(self, total_epochs: int):
                super().__init__()
                self.total_epochs = total_epochs
                self.start_time = None

            def on_train_begin(self, logs=None):
                self.start_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                elapsed = time.time() - self.start_time
                avg_epoch_time = elapsed / (epoch + 1)
                eta = avg_epoch_time * (self.total_epochs - (epoch + 1))
                epoch_data = {
                    "epoch": epoch + 1,
                    "total_epochs": self.total_epochs,
                    "elapsed": elapsed,
                    "eta": eta,
                    "status": "running",
                    "loss": float(logs.get("loss", 0.0)),
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "acc": float(logs.get("accuracy", 0.0)),
                    "val_acc": float(logs.get("val_accuracy", 0.0)),
                    "precision": float(logs.get("precision", 0.0)),
                    "recall": float(logs.get("recall", 0.0)),
                    "val_precision": float(logs.get("val_precision", 0.0)),
                    "val_recall": float(logs.get("val_recall", 0.0)),
                    "auc": float(logs.get("auc", 0.0)),
                    "val_auc": float(logs.get("val_auc", 0.0)),
                    "f1_score": float(logs.get("f1_score", 0.0)),
                    "val_f1_score": float(logs.get("val_f1_score", 0.0)),
                    "message": f"Epoch {epoch + 1}/{self.total_epochs} completed"
                }
                update_progress(**epoch_data)


            def on_train_end(self, logs=None):
                total_time = time.time() - self.start_time
                update_progress(status="success", message="Training complete", elapsed=total_time)

        class PathTrackingModelCheckpoint(ModelCheckpoint):
            """
            A custom ModelCheckpoint to track the path of the best saved model file.
            """
            def __init__(self, classifier_instance, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.classifier_instance = classifier_instance

            def on_epoch_end(self, epoch, logs=None):
                super().on_epoch_end(epoch, logs)
                
                # Check if the model was saved (i.e., if it was the best so far)
                if self.save_best_only and self.best == logs.get(self.monitor):
                    # The saved filepath is guaranteed to be correct due to static naming
                    self.classifier_instance.best_checkpoint_path = self.filepath
                    logger.info(f"[CHECKPOINT] Updated best path: {self.filepath}")

        # 1. Setup
        steps_per_epoch, val_steps, test_steps = self._setup_data_generators(batch_size=batch_size)
        self._build_model()
        compile_model()

        # 2. Callbacks
        callbacks = get_callbacks()
        callbacks.append(TrainingProgressCallback(total_epochs=epochs))

        # 3. Class weights
        class_weight_dict = get_class_weights()
        logger.info(f"[TRAIN] Using class weights: {class_weight_dict}")
        
        # 4. Train
        logger.info("[TRAIN] Entering training loop...")
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=val_steps,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # 5. Evaluation
        results = evaluate_training()
        logger.info(f"[TRAIN] Training complete in {results['training_time']:.2f} seconds.")
        logger.info(f"[TRAIN] Test Results: Loss={results.get('loss'):.4f}, Accuracy={results.get('accuracy'):.4f}")
        return results
    
    def save_history(self, file_path: Path = RESULTS_DIR / "training_history.json") -> bool:
        """
        Saves the training history to a JSON file.

        :param file_path: Path to save the history JSON file.
        :type file_path: str
        :returns: True on success, False otherwise.
        :rtype: bool
        """
        if self.history is None:
            logger.warning("[HISTORY] No training history available to save.")
            return False
        
        # Convert history to pure Python types
        history_dict = {"history": {k: [float(vv) for vv in v] for k, v in self.history.history.items()}}

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(history_dict, f, indent=4)
            logger.info(f"[HISTORY] Training history saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"[HISTORY] Error saving training history: {e}")
            return False

    def load_history(self, file_path: Path = RESULTS_DIR / "training_history.json") -> bool:
        """
        Loads training history from a JSON file.
        
        :param file_path: Path to load the history JSON file.
        :type file_path: str
        :returns: True on success, False otherwise.
        :rtype: bool
        """
        if not os.path.exists(file_path):
            logger.warning(f"[HISTORY] History file does not exist: {file_path}")
            return None

        class History:
            def __init__(self, history_dict):
                self.history = history_dict
                
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.history = History(data["history"])
            logger.info("[HISTORY] History loaded succesfully")
            return True
        except Exception as e:
            logger.error(f"[HISTORY] Error loading training history: {e}")
            return False
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict the class for a single input image.
        
        :param image_path: Path to the image file to classify.
        :type image_path: str
        :returns: (Predicted class name, confidence score, dictionary of all class probabilities).
        :rtype: Tuple[str, float, Dict[str, float]]
        """
        logger.info(f"[PREDICT] Making prediction for image: {image_path}")
        
        if self.model is None:
            logger.error("[PREDICT] Model not trained or loaded.")
            raise ValueError("Model not trained or loaded")
        if not os.path.exists(image_path):
            logger.error(f"[PREDICT] Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array / 255.0, axis=0)
        except Exception as e:
            logger.error(f"[PREDICT] Error loading/preprocessing image: {e}")
            raise

        probs = self.model.predict(img_array, verbose=0)[0]

        if self.num_classes == 2:
            probs = np.array([1 - probs[0], probs[0]])  # force shape (2,)
        
        pred_idx = int(np.argmax(probs))
        predicted_class = self.class_labels[pred_idx]
        confidence = float(probs[pred_idx])
        probabilities = {label: float(p) for label, p in zip(self.class_labels, probs)}

        logger.info(f"[PREDICT] Result: Predicted class: {predicted_class}, Confidence: {confidence:.2%}")
        return predicted_class, confidence, probabilities
    
    def evaluate_model(self, return_details: bool = False):
        """
        Evaluate the model on the validation set.

        :param return_details: If True, returns raw predictions and labels alongside metrics.
        :type return_details: bool
        :returns: Dictionary containing evaluation metrics (and optional details).
        :rtype: Dict
        """
        if self.val_generator is None:
            logger.warning("[PREDICT] val_generator is None. Preparing data...")
            self._setup_data_generators()

        # Standard Keras metrics (loss, acc, precision, recall, etc.)
        keras_results = self.model.evaluate(self.val_generator, verbose=0, return_dict=True)

        # Predictions for confusion-matrix-based metrics
        probs = self.model.predict(self.val_generator, verbose=0)
        if probs.shape[1] == 1:  # binary -> convert to (n_samples, 2)
            probs = np.hstack([1 - probs, probs])

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        labels = self.val_generator.classes

        sensitivity = specificity = None
        if self.num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Merge results
        summary = {
            "model": self.model_name,
            "num_classes": self.num_classes,
            "loss": keras_results.get("loss"),
            "accuracy": keras_results.get("accuracy"),
            "precision": keras_results.get("precision"),
            "recall": keras_results.get("recall"),
            "auc": keras_results.get("auc"),
            "f1_score": keras_results.get("f1_score"),
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

        if return_details:
            summary.update({
                "class_labels": self.class_labels,
                "probs": probs,
                "pred_indices": preds,
                "confidences": confs,
                "true_indices": labels,
            })
        
        logger.info("[PREDICT] Model evaluation on validation set complete.")
        return summary
    
    #endregion
    
    
    #region Generating plotts/heatmaps
    
    def generate_gradcam_heatmap(self, 
                                 image_path: str, 
                                 conv_layer_name: Optional[str] = None, 
                                 alpha: float = 0.5,
                                 output_dir: Path = GRADCAM_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap for model interpretability on a single image.
        
        :param image_path: Path to the input image file.
        :type image_path: str
        :param conv_layer_name: Name of the convolutional layer to analyze. Uses the last Conv2D layer if None.
        :type conv_layer_name: Optional[str]
        :param alpha: Transparency factor for heatmap overlay (0.0-1.0).
        :type alpha: float
        :param output_dir: Directory to save the generated heatmap image.
        :type output_dir: str
        :returns: (Original image with heatmap overlay, raw heatmap visualization, original preprocessed image).
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        :raises ValueError: If the model is not trained/loaded.
        :raises FileNotFoundError: If the image file doesn't exist.
        """
        logger.info(f"[GRADCAM] Generating Grad-CAM for image: {image_path}")
        logger.info(f"  Using model: {self.model_name}, Image size: {self.img_size}")        
        
        if self.model is None:
            logger.error("[GRADCAM] Model has not been trained or loaded.")
            raise ValueError("Model has not been trained or loaded yet.")
        if not os.path.exists(image_path):
            logger.error(f"[GRADCAM] Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        os.makedirs(output_dir, exist_ok=True)
        
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array / 255.0, axis=0)
        except Exception as e:
            logger.error(f"[GRADCAM] Error loading/preprocessing image: {e}")
            raise

        # Find the last convolutional layer if not specified
        if conv_layer_name is None:
            conv_layers = []
            for layer in self.model.layers:
                if isinstance(layer, Conv2D):
                    conv_layers.append(layer.name)
            
            if not conv_layers:
                logger.warning("[GRADCAM] Could not find any Conv2D layers in the model.")
                return None, None, None
            
            conv_layer_name = conv_layers[-1]  # Use the last conv layer
            logger.info(f"  Analyzing convolutional layer: {conv_layer_name}")
                
        try:
            # Create Grad-CAM model
            grad_model = Model(
                inputs=self.model.inputs,
                outputs=[self.model.get_layer(conv_layer_name).output, self.model.output]
            )

            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if self.num_classes == 2:
                    # Binary classification - use the sigmoid output directly
                    class_output = predictions[0][0]
                else:
                    # Multi-class classification - use the highest probability class
                    predicted_class_idx = tf.argmax(predictions[0])
                    class_output = predictions[0][predicted_class_idx]
                
            # Calculate gradients of loss with respect to conv layer output
            grads = tape.gradient(class_output, conv_outputs)

            if grads is None:
                logger.error(f"[GRADCAM] Gradient is None for layer '{conv_layer_name}'. Check layer existence.")
                raise ValueError(
                    f"Gradient is None for layer '{conv_layer_name}'. "
                    f"Check that the layer exists and is differentiable."
                )

            # Pool gradients over spatial dimensions (Global Average Pooling of gradients)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature maps by the gradients
            conv_outputs = conv_outputs[0]  # Remove batch dimension
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Apply ReLU to focus on features that positively influence the prediction
            heatmap = tf.nn.relu(heatmap)
            
            # Normalize heatmap to [0, 1]
            heatmap_max = tf.reduce_max(heatmap)
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            
            # Convert to numpy and resize to original image dimensions
            heatmap = heatmap.numpy()
            heatmap = cv2.resize(heatmap, self.img_size)
            
            # Convert to colormap for visualization
            heatmap_colored = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

            # Load original image and superimpose heatmap
            original_img = cv2.imread(image_path)
            if original_img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            original_img = cv2.resize(original_img, self.img_size)
            superimposed_img = cv2.addWeighted(original_img, 1.0 - alpha, heatmap_colored, alpha, 0)
            
            # Save the results
            
            # base model-specific directory
            timestamp = datetime.now().strftime("%Y%m%d")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_output_dir = output_dir / f"{image_name}_{self.model_name}_{timestamp}"
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            # save components
            cv2.imwrite(str(image_output_dir / "original.png"), original_img)
            cv2.imwrite(str(image_output_dir / f"{conv_layer_name}_heatmap.png"), heatmap_colored)
            cv2.imwrite(str(image_output_dir / f"{conv_layer_name}_gradcam.png"), superimposed_img)
            
            logger.info(f"  Grad-CAM results saved to: {image_output_dir}")
                        
            return original_img, heatmap_colored, superimposed_img
        
        except Exception as e:
            logger.error(f"[GRADCAM] Error generating Grad-CAM heatmap: {e}")
            raise
    
    def generate_gradcam_heatmap_multiple_layers(self, 
                                                 image_path: str, 
                                                 alpha: float = 0.5, 
                                                 output_dir: Path = GRADCAM_DIR) -> None:
        """
        Generates and saves a grid of Grad-CAM heatmaps from multiple convolutional layers.
        
        :param image_path: Path to the input image file.
        :type image_path: str
        :param alpha: Transparency factor for heatmap overlay (0.0-1.0).
        :type alpha: float
        :param output_dir: Directory to save the layer comparison grid and individual heatmaps.
        :type output_dir: str
        :raises ValueError: If the model is not trained/loaded or no conv layers are found.
        :raises FileNotFoundError: If the image file doesn't exist.
        """
        logger.info(f"[GRADCAM] Starting multi-layer Grad-CAM test for image: {image_path}")
        
        if self.model is None:
            logger.error("[GRADCAM] Model has not been trained or loaded.")
            raise ValueError("Model has not been trained or loaded yet.")
        if not os.path.exists(image_path):
            logger.error(f"[GRADCAM] Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Find all convolutional layers
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, Conv2D):
                conv_layers.append(layer.name)
        if not conv_layers:
            logger.error("[GRADCAM] No convolutional layers found in the model.")
            raise ValueError("No convolutional layers found in the model.")
        logger.info(f"  Found {len(conv_layers)} convolutional layers. Analyzing...")    
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = output_dir / f"{image_name}_{self.model_name}_{timestamp}"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate heatmaps for each layer
        results = {}
        valid_layers = []
        
        for layer_name in conv_layers:
            try:
                original_img, heatmap, superimposed_img = self.generate_gradcam_heatmap(
                    image_path, 
                    conv_layer_name=layer_name, 
                    alpha=alpha,
                )
                
                results[layer_name] = {
                    'superimposed': superimposed_img,
                    'heatmap': heatmap,
                    'original': original_img
                }
                valid_layers.append(layer_name)
                
            except Exception as e:
                logger.warning(f"[GRADCAM] Error generating Grad-CAM for layer {layer_name}: {e}")
                continue
        
        if not valid_layers:
            logger.error("[GRADCAM] Could not generate Grad-CAM for any layers.")
            raise ValueError("Could not generate Grad-CAM for any layers.")
        
        # Create comparison grid
        try:
            n_layers = len(valid_layers)
            if n_layers <= 4:
                rows, cols = 2, 2
            elif n_layers <= 6:
                rows, cols = 2, 3
            elif n_layers <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, 4
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if n_layers == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, layer_name in enumerate(valid_layers):
                if i < len(axes):
                    superimposed_img = results[layer_name]['superimposed']
                    # Convert BGR to RGB for matplotlib
                    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
                    
                    axes[i].imshow(superimposed_img_rgb)
                    axes[i].set_title(f'Layer: {layer_name}', fontsize=12)
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(valid_layers), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            comparison_path = image_output_dir / f"comparison_{image_name}.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[GRADCAM] Layer comparison grid saved to: {comparison_path}")
            
        except Exception as e:
            logger.warning(f"[GRADCAM] Error creating comparison grid: {e}")
        
        logger.info(f"[GRADCAM] Multi-layer analysis completed. Results saved in: {output_dir}")
    
    
    def _save_plot(self, plt, file_name: str, save_path: Path = RESULTS_DIR):
        """
        Helper function to save a matplotlib plot to a file.
        
        :param plt: The matplotlib plot object.
        :type plt: module
        :param file_name: Name of the file to save (e.g., 'plot.png').
        :type file_name: str
        :param save_path: Directory path where the plot should be saved.
        :type save_path: str
        :returns: The absolute path to the saved plot file.
        :rtype: str
        """
        save_path.mkdir(parents=True, exist_ok=True)
        path = save_path / file_name
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"[PLOT] Plot saved to {path}")
        return path
    
    def plot_confusion_matrix(self, normalize: bool = False, save_path: Path = RESULTS_DIR):
        """
        Plots and saves the confusion matrix for the test set predictions.

        :param normalize: If True, normalize the matrix to show percentages.
        :type normalize: bool
        :param save_path: Directory path where the plot should be saved.
        :type save_path: str
        """
        if self.model is None or self.class_labels is None or self.num_classes is None:
            logger.warning("[PLOT] Required components (model/classes) not available for confusion matrix.")
            return

        if self.val_generator is None:
            self._setup_data_generators()

        # Calculate
        labels = list(self.class_labels.values())
        y_true = self.val_generator.classes
        y_pred = self.model.predict(self.val_generator, verbose=0)

        if self.num_classes == 2:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        logger.info(f"Calculated confusion matrix (normalize={normalize}):\n{cm}")
        logger.info(f"Accuracy: {np.trace(cm) / np.sum(cm):.4f}")

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            square=True,
        )
        plt.title("Confusion Matrix", fontsize=16, weight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_results_dir = save_path / 'confusion_matrix' / f'{self.model_name}_{timestamp}'
        self._save_plot(plt, "confusion_matrix.png", model_results_dir)

    def plot_data_distribution(self, save_path: Path = RESULTS_DIR):
        """
        Plots and saves the distribution of data across train, validation, and test sets.

        :param save_path: Directory path where the plot should be saved.
        :type save_path: str
        """
        if self.class_labels is None:
            logger.warning("[PLOT] Required class labels not available for data distribution plot.")
            return
        
        if self.test_generator is None or self.val_generator is None or self.train_generator is None:
            self._setup_data_generators()
        
        # Calculate
        labels = list(self.class_labels.values())
        datasets = {
            "Training": self.train_generator.classes,
            "Validation": self.val_generator.classes,
            "Test": self.test_generator.classes,
        }
        class_counts = {
            name: [np.sum(np.array(data) == idx) for idx in self.class_labels.keys()]
            for name, data in datasets.items()
        }
        

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bottom = np.zeros(len(datasets))
        for i, class_name in enumerate(labels):
            counts = [class_counts[d_name][i] for d_name in datasets.keys()]
            p = ax.bar(datasets.keys(), counts, label=class_name, bottom=bottom)
            bottom += counts
            ax.bar_label(p, label_type='center', color='white', weight='bold')

        ax.set_title("Data Distribution Across Sets", fontsize=16, weight="bold")
        ax.set_ylabel("Number of Samples")
        ax.legend(title="Class")
        plt.tight_layout()
        
        self._save_plot(plt, "data_distribution_bar.png", save_path)

    def plot_images_example(self, save_path: Path = RESULTS_DIR):
        """
        Saves a grid of example images from the training set with their labels.

        :param save_path: Directory path where the plot should be saved.
        :type save_path: str
        """
        if self.class_labels is None or self.num_classes is None:
            logger.warning("[PLOT] Required components (train data/classes) not available for example images plot.")
            return
        
        if self.train_generator is None:
            self._setup_data_generators()
        
        images, labels = next(self.train_generator)
        
        # Plotting
        plt.figure(figsize=(12, 12))
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            label_idx = np.argmax(labels[i]) if self.num_classes > 2 else int(labels[i])
            plt.title(self.class_labels.get(label_idx, "Unknown"))
            plt.axis("off")
        plt.tight_layout()
        
        self._save_plot(plt, "example_images.png", save_path)
    
    def plot_model_architecture(self, save_path: Path = RESULTS_DIR):
        """
        Saves a visualization of the model architecture to a file using Keras utils.

        :param save_path: Directory path where the plot should be saved.
        :type save_path: str
        """
        if self.model is None:
            logger.warning("[PLOT] Model not available for plotting model architecture.")
            return
        
        # Plotting
        path = os.path.join(save_path, f"Architecture_{self.model_name}_.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tf.keras.utils.plot_model(
            self.model,
            to_file=path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96
        )
        logger.info(f"[PLOT] Model architecture saved to {path}")
    
    def _plot_metric(self, ax, epochs, history, metric_name, title):
        """
        Helper function to plot a single metric for train and validation sets.

        :param ax: Matplotlib axes object to draw on.
        :param epochs: Array/list of epoch numbers.
        :param history: Dictionary of training history.
        :param metric_name: Name of the metric (e.g., 'loss', 'accuracy', 'val_loss').
        :param title: Title for the plot.
        """
        train_metric = history.get(metric_name)
        val_metric = history.get(f"val_{metric_name}")
        
        if train_metric is not None:
            ax.plot(epochs, train_metric, "b-o", label=f"Train {title}")
        if val_metric is not None:
            ax.plot(epochs, val_metric, "r-o", label=f"Validation {title}")

        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.6)
        if train_metric is not None or val_metric is not None:
            ax.legend()
    
    def plot_training_history(self, save_path: Path = RESULTS_DIR):
        """
        Plots and saves the model's training and validation history for key metrics (loss, accuracy, etc.).

        :param save_path: Directory path where the plot should be saved.
        :type save_path: str
        """
        if self.history is None:
            logger.warning("[PLOT] Training history not available for plotting history.")
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Plotting
        metrics_to_plot = {
            "Accuracy": "accuracy",
            "Loss": "loss",
            "AUC": "auc",
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.flatten()

        # Use the helper to plot each metric
        for i, (title, metric_name) in enumerate(metrics_to_plot.items()):
            self._plot_metric(axes[i], epochs, history, metric_name, title)
            
        # Turn off the last unused subplot if there's an odd number of plots
        if len(metrics_to_plot) < len(axes):
            for i in range(len(metrics_to_plot), len(axes)):
                axes[i].axis('off')

        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_results_dir = save_path / 'training_history' / f'{self.model_name}_{timestamp}'
        self._save_plot(plt, "training_history.png", model_results_dir)

    #endregion


def ensemble_testing():
    data_dir = DATA_DIR
    
    checkpoint_paths = {
        "OwnV3":   f"{CHECKPOINT_DIR}/Saved/OwnV3.epoch50-val_acc0.9830.hdf5",
        "OwnV2":   f"{CHECKPOINT_DIR}/Saved/OwnV2.epoch28-val_acc0.9705.hdf5",
        "OwnV1":   f"{CHECKPOINT_DIR}/Saved/OwnV1.epoch26-val_acc0.9761.hdf5",
        "VGG16":   f"{CHECKPOINT_DIR}/Saved/VGG16.epoch18-val_acc0.9534.hdf5",
        "AlexNet": f"{CHECKPOINT_DIR}/Saved/AlexNet.epoch27-val_acc0.9761.hdf5"
    }
    
    weights = []
    for path in checkpoint_paths.values():
        val_acc_str = path.split("val_acc")[-1].split(".hdf5")[0]
        weights.append(float(val_acc_str))
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    logger.info(f"Using ensemble weights: {weights}")
    
    def evaluate_ensemble(models_info, data_dir, img_size=(150, 150), weights=None):
        """
        Evaluate an ensemble of models on the same dataset.
        Returns: dict with ensemble accuracy, predictions, and individual results.
        """
        all_probs = []
        true_labels = None
        individual_results = []

        # Load models and collect predictions
        for name, ckpt in models_info.items():
            clf = Classifier(model_name=name, img_size=img_size, data_dir=data_dir)
            clf.load_model(ckpt)
            res = clf.evaluate_model(return_details=True)
            
            # Save per-model results
            acc = np.mean(np.argmax(res["probs"], axis=1) == res["true_indices"])
            individual_results.append({"model": name, "accuracy": acc})
            logger.info(f"  {name} individual accuracy: {acc:.4f}")
            logger.info(f"  Auto results: {res}")            

            all_probs.append(res["probs"])
            if true_labels is None:
                true_labels = res["true_indices"]
        
        if weights is None:
            weights = np.ones(len(all_probs)) / len(all_probs)

        # Calculate the weighted average of the probabilities
        avg_probs = np.average(np.stack(all_probs, axis=0), axis=0, weights=weights)
        preds = np.argmax(avg_probs, axis=1)
        ensemble_acc = np.mean(preds == true_labels)
        
        return {
            "ensemble_accuracy": ensemble_acc,
            "avg_probs": avg_probs,
            "preds": preds,
            "true_labels": true_labels,
            "individual_results": individual_results
        }
    
    results = evaluate_ensemble(checkpoint_paths, data_dir, (150, 150), weights=weights)
    logger.info("=== Ensemble Evaluation Results ===")
    logger.info(f"Ensemble accuracy: {results['ensemble_accuracy']:.4f}")
    for r in results["individual_results"]:
        logger.info(f"  {r['model']}: {r['accuracy']:.4f}")

def model_testing():
    data_dir = DATA_DIR
    models = [
        'OwnV4', 'OwnV3', 'OwnV2', 'OwnV1', 
        'AlexNet', 'VGG16', 'VGG19', 'ResNet50', 'DenseNet121', 'MobileNetV2',
    ]
    
    classifier = Classifier(
        model_name='OwnV3',
        img_size=(150, 150),
        # img_size=(224, 224),
        data_dir=data_dir
    )
    
    # Train the model
    # results = classifier.train(
    #     epochs=70,
    #     batch_size=16, 
    #     learning_rate=0.0003,
    # )
    # classifier.save_history()
    # logger.info(results)
    
    # if classifier.model is None:
    #     classifier.load_model(f"{CHECKPOINT_DIR}/Saved/OwnV3.epoch50-val_acc0.9830.hdf5")
    # if classifier.history is None:
    #     classifier.load_history()
    
    
    # Plotting
    # classifier.plot_confusion_matrix()
    # classifier.plot_training_history()
    classifier.plot_data_distribution()
    # classifier.plot_images_example()
    # classifier.plot_model_architecture()
    
    
    # Evaluate model
    # results = classifier.evaluate_model()
    # logger.info("Evaluation results:")
    # for k, v in results.items():
    #     if isinstance(v, (float, int)) and v is not None:
    #         logger.info(f"  {k}: {v:.4f}")
    
    
    # Image prediction / heatmap testing
    #             backend/api/classifier/data/train/PNEUMONIA/PNEUMONIA_person1929_bacteria_4839_aug1_rot14.8_bright0.88_cont1.08_blur_color0.94.jpeg
    # OwnV3only - backend/api/classifier/data/train/PNEUMONIA/PNEUMONIA_person1921_bacteria_4828_aug1_rot-6.4_hflip_bright0.83_cont1.18_blur.jpeg
    # GoodMap   - backend/api/classifier/data/train/PNEUMONIA/PNEUMONIA_person418_virus_852_aug1_rot11.9_hflip_bright0.98_color0.91.jpeg
    
    # image_to_test = f'{DATA_DIR}/train/PNEUMONIA/PNEUMONIA_person1929_bacteria_4839_aug1_rot14.8_bright0.88_cont1.08_blur_color0.94.jpeg'
    # predicted_class, confidence, _ = classifier.predict(image_to_test)
    
    # classifier.generate_gradcam_heatmap(image_to_test)
    # classifier.generate_gradcam_heatmap_multiple_layers(image_to_test)


if __name__ == "__main__":
    # ensemble_testing()
    model_testing()
    
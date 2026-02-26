#!/usr/bin/env python3
"""
RadioML Signal Classification - Unified Training Script
Supports both RML2016.10a and RML2016.10b datasets.

Usage examples:
python main.py --models resnet cnn1d --mode train
python main.py --models hybrid_complex_resnet --mode evaluate
python main.py --models cnn2d complex_nn --mode all --epochs 100
python main.py --dataset 2016b --models resnet --mode train
"""

import os
import argparse
import time
import random
import numpy as np
import tensorflow as tf
import torch

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all logs, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
tf.get_logger().setLevel('ERROR')

# Import project modules
from explore_dataset import load_radioml_data, explore_dataset, plot_signal_examples
from preprocess import prepare_data_by_snr_stratified
from train import train_model, train_torch_model, plot_training_history
from models import (
    build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model,
    get_callbacks,
    build_hybrid_complex_resnet_model, build_lightweight_hybrid_model,
    # AMC-Net model
    build_amcnet_model,
    # Benchmark models
    build_cldnn_model, build_cgdnn_model, build_dae_model,
    # ULCNN models
    build_mcldnn_model, build_scnn_model, build_mcnet_model, build_pet_model, build_ulcnn_model,
    # New ultra-lightweight hybrid models
    build_ultra_lightweight_hybrid_model, build_micro_lightweight_hybrid_model,
    # PyTorch models
    build_iqformer_model, build_fea_t_model
)
from evaluate import evaluate_by_snr, evaluate_torch_by_snr
from model.custom_objects import get_custom_objects_for_model


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    '2016a': {
        'dataset_path': '../data/RML2016.10a_dict.pkl',
        'output_dir': '../output',
        'denoised_cache_dir': '../denoised_datasets',
        'file_suffix_prefix': '',
        'dataset_label': 'RML2016.10a',
    },
    '2016b': {
        'dataset_path': '../data/RML2016.10b.dat',
        'output_dir': '../output_2016b',
        'denoised_cache_dir': '../denoised_datasets_2016b',
        'file_suffix_prefix': '_2016b',
        'dataset_label': 'RML2016.10b',
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_random_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print(f"Random seed set to {seed}")


def configure_gpu(gpu_id=None):
    """Configure GPU devices for training"""
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("No GPUs found. Using CPU for training.")
        return

    print(f"Available GPUs: {[gpu.name for gpu in gpus]}")

    if gpu_id is None:
        # Use all available GPUs
        print("Using all available GPUs")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        # Use specific GPU(s)
        try:
            gpu_ids = [int(x.strip()) for x in gpu_id.split(',')]
            selected_gpus = [gpus[i] for i in gpu_ids if i < len(gpus)]

            if not selected_gpus:
                print(f"Invalid GPU ID(s): {gpu_id}. Using all available GPUs.")
                selected_gpus = gpus

            # Set visible devices
            tf.config.set_visible_devices(selected_gpus, 'GPU')
            print(f"Using GPU(s): {[gpu.name for gpu in selected_gpus]}")

            # Enable memory growth for selected GPUs
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except (ValueError, IndexError) as e:
            print(f"Error parsing GPU ID '{gpu_id}': {e}")
            print("Using all available GPUs")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print("Note: GPU configuration must be set before creating any operations")


def get_file_suffix(denoising_method, augment_data, dataset_suffix_prefix=''):
    """Generate suffix for file names based on denoising method, data augmentation, and dataset type"""
    suffix = dataset_suffix_prefix
    if denoising_method and denoising_method != 'none':
        suffix += f"_{denoising_method}"
    if augment_data:
        suffix += "_augment"
    return suffix


def prepare_data_with_cache_suffix(dataset, cache_filename_suffix='', **kwargs):
    """Wrapper for prepare_data_by_snr_stratified that adds a suffix to denoised cache filenames.

    For the 2016b dataset, cached denoised files get a '_2016b' suffix so they don't collide
    with 2016a cache files.
    """
    if not cache_filename_suffix:
        return prepare_data_by_snr_stratified(dataset, **kwargs)

    import preprocess

    original_create_filename = preprocess.create_denoised_filename

    def create_denoised_filename_with_suffix(denoising_method, specific_snrs=None):
        original_filename = original_create_filename(denoising_method, specific_snrs)
        if original_filename.endswith('.pkl'):
            return original_filename[:-4] + cache_filename_suffix + '.pkl'
        else:
            return original_filename + cache_filename_suffix

    preprocess.create_denoised_filename = create_denoised_filename_with_suffix

    try:
        result = prepare_data_by_snr_stratified(dataset, **kwargs)
        return result
    finally:
        preprocess.create_denoised_filename = original_create_filename


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_available_models():
    """Return list of all available model types"""
    return [
        'cnn1d', 'cnn2d', 'resnet', 'complex_nn',
        'hybrid_complex_resnet', 'lightweight_hybrid',
        # AMC-Net model
        'amcnet',
        # Benchmark models
        'cldnn', 'cgdnn', 'dae',
        # ULCNN models
        'mcldnn', 'scnn', 'mcnet', 'pet', 'ulcnn',
        # New ultra-lightweight hybrid models
        'ultra_lightweight_hybrid', 'micro_lightweight_hybrid',
        # PyTorch models
        'iqformer', 'fea_t'
    ]


def expand_model_selection(selected_models):
    """Expand model selection to handle special cases.
    Returns a list of individual model names to process.
    """
    expanded_models = []
    for model in selected_models:
        expanded_models.append(model)
    return expanded_models


def build_model_by_name(model_name, input_shape, num_classes):
    """Build a model by name and return the model instance"""
    model_builders = {
        'cnn1d': build_cnn1d_model,
        'cnn2d': build_cnn2d_model,
        'resnet': build_resnet_model,
        'complex_nn': build_complex_nn_model,
        'hybrid_complex_resnet': build_hybrid_complex_resnet_model,
        'lightweight_hybrid': build_lightweight_hybrid_model,
        # AMC-Net model
        'amcnet': build_amcnet_model,
        # Benchmark models
        'cldnn': build_cldnn_model,
        'cgdnn': build_cgdnn_model,
        'dae': build_dae_model,
        # ULCNN models
        'mcldnn': build_mcldnn_model,
        'scnn': build_scnn_model,
        'mcnet': build_mcnet_model,
        'pet': build_pet_model,
        'ulcnn': build_ulcnn_model,
        # New ultra-lightweight hybrid models
        'ultra_lightweight_hybrid': build_ultra_lightweight_hybrid_model,
        'micro_lightweight_hybrid': build_micro_lightweight_hybrid_model,
        # PyTorch models
        'iqformer': build_iqformer_model,
        'fea_t': build_fea_t_model,
    }

    if model_name in model_builders:
        return model_builders[model_name](input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def is_torch_model_name(model_name):
    """Whether the model is trained/evaluated with PyTorch pipeline."""
    return model_name in {'iqformer', 'fea_t'}


def validate_model_selection(selected_models):
    """Validate that all selected models are available"""
    available_models = get_available_models()
    invalid_models = [m for m in selected_models if m not in available_models]

    if invalid_models:
        print(f"Error: Invalid model(s) selected: {invalid_models}")
        print(f"Available models: {available_models}")
        return False

    return True


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_selected_models(selected_models, X_train, y_train, X_val, y_val, input_shape, num_classes,
                          models_dir, plots_dir, suffix, batch_size, epochs):
    """Train all selected models"""

    for model_name in selected_models:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model...")
        print(f"{'='*60}")

        try:
            # Handle special data preparation for CNN2D
            if model_name == 'cnn2d':
                X_train_model = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
                X_val_model = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
            else:
                X_train_model = X_train
                X_val_model = X_val

            # Build model
            model = build_model_by_name(model_name, input_shape, num_classes)

            print(f"Model architecture for {model_name}:")
            if is_torch_model_name(model_name):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(model)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
            else:
                model.summary()

            # Train model
            if is_torch_model_name(model_name):
                history = train_torch_model(
                    model,
                    X_train_model, y_train,
                    X_val_model, y_val,
                    os.path.join(models_dir, f"{model_name}_model{suffix}.pt"),
                    batch_size=batch_size,
                    epochs=epochs
                )
            else:
                history = train_model(
                    model,
                    X_train_model, y_train,
                    X_val_model, y_val,
                    os.path.join(models_dir, f"{model_name}_model{suffix}.keras"),
                    batch_size=batch_size,
                    epochs=epochs
                )

            # Plot and save training history
            plot_training_history(
                history,
                os.path.join(plots_dir, f"{model_name}_training_history{suffix}.png")
            )

            print(f"Successfully completed training for {model_name}")

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue


def evaluate_model_variants(model_name, model_base_path, X_test, y_test, snr_test, mods,
                            results_dir, input_shape, num_classes, suffix="", custom_objects=None):
    """Evaluate both the best model and the last epoch model for a given model type.

    Args:
        model_name: Name of the model type (e.g., 'cnn1d', 'resnet')
        model_base_path: Base path without extension (e.g., '/path/models/cnn1d_model')
        X_test, y_test, snr_test, mods: Test data and metadata
        results_dir: Directory to save evaluation results
        suffix: File suffix for distinguishing configurations
        custom_objects: Custom objects needed for model loading (optional)
    """
    # Standard Keras/PyTorch model evaluation
    model_ext = ".pt" if is_torch_model_name(model_name) else ".keras"
    best_model_path = model_base_path + model_ext
    if os.path.exists(best_model_path):
        print(f"\nEvaluating {model_name} Model (Best)...")
        try:
            if is_torch_model_name(model_name):
                best_model = build_model_by_name(model_name, input_shape, num_classes)
                checkpoint = torch.load(best_model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    best_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    best_model.load_state_dict(checkpoint)
                evaluate_torch_by_snr(
                    best_model,
                    X_test, y_test, snr_test, mods,
                    os.path.join(results_dir, f'{model_name}_evaluation_results{suffix}')
                )
            else:
                if custom_objects:
                    best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
                else:
                    best_model = tf.keras.models.load_model(best_model_path)
                evaluate_by_snr(
                    best_model,
                    X_test, y_test, snr_test, mods,
                    os.path.join(results_dir, f'{model_name}_evaluation_results{suffix}')
                )
            print(f"Successfully loaded best model from {best_model_path}")
        except Exception as e:
            print(f"Error loading or evaluating best model {best_model_path}: {e}")
    else:
        print(f"Best model {best_model_path} not found for evaluation.")

    # Evaluate last epoch model
    last_model_path = model_base_path + "_last" + model_ext
    if os.path.exists(last_model_path):
        print(f"\nEvaluating {model_name} Model (Last Epoch)...")
        try:
            if is_torch_model_name(model_name):
                last_model = build_model_by_name(model_name, input_shape, num_classes)
                checkpoint = torch.load(last_model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    last_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    last_model.load_state_dict(checkpoint)
                evaluate_torch_by_snr(
                    last_model,
                    X_test, y_test, snr_test, mods,
                    os.path.join(results_dir, f'{model_name}_evaluation_results_last{suffix}')
                )
            else:
                if custom_objects:
                    last_model = tf.keras.models.load_model(last_model_path, custom_objects=custom_objects)
                else:
                    last_model = tf.keras.models.load_model(last_model_path)
                evaluate_by_snr(
                    last_model,
                    X_test, y_test, snr_test, mods,
                    os.path.join(results_dir, f'{model_name}_evaluation_results_last{suffix}')
                )
            print(f"Successfully loaded last epoch model from {last_model_path}")
        except Exception as e:
            print(f"Error loading or evaluating last model {last_model_path}: {e}")
    else:
        print(f"Last model {last_model_path} not found for evaluation.")


def evaluate_selected_models(selected_models, X_test, y_test, snr_test, mods,
                             models_dir, results_dir, suffix):
    """Evaluate all selected models"""

    # Enable unsafe deserialization for models with custom layers
    tf.keras.config.enable_unsafe_deserialization()

    for model_name in selected_models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} Model...")
        print(f"{'='*60}")

        try:
            custom_objects = get_custom_objects_for_model(model_name)
            model_base_path = os.path.join(models_dir, f"{model_name}_model{suffix}")
            input_shape = X_test.shape[1:]
            num_classes = y_test.shape[1]

            evaluate_model_variants(
                model_name, model_base_path, X_test, y_test, snr_test, mods,
                results_dir, input_shape, num_classes, suffix, custom_objects
            )

            print(f"Successfully completed evaluation for {model_name}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='RadioML Signal Classification with Flexible Model Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models resnet cnn1d --mode train
  %(prog)s --models hybrid_complex_resnet --mode evaluate
  %(prog)s --models cnn2d complex_nn --mode all --epochs 100
  %(prog)s --dataset 2016b --models resnet --mode train
  %(prog)s --models cnn1d resnet --mode train --gpu_id 0
  %(prog)s --models resnet --mode train --gpu_id 0,1
        """
    )

    parser.add_argument('--dataset', type=str, default='2016a',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset version to use (default: 2016a)')

    parser.add_argument('--mode', type=str, default='all',
                        choices=['explore', 'train', 'evaluate', 'all'],
                        help='Mode of operation')

    parser.add_argument('--models', type=str, nargs='+',
                        choices=get_available_models(),
                        required=True,
                        help='Model architectures to use (select multiple)')

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the RadioML dataset (overrides dataset default)')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for outputs (overrides dataset default)')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')

    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--augment_data', action='store_true',
                        help='Enable data augmentation for training data (11 rotations, 30 deg increments)')

    parser.add_argument('--denoising_method', type=str, default='efficient_gpr_per_sample',
                        choices=['gpr', 'efficient_gpr_per_sample', 'gpr_fft', 'none'],
                        help='Denoising method to apply to the signals')

    parser.add_argument('--denoised_cache_dir', type=str, default=None,
                        help='Directory to save/load cached denoised datasets (overrides dataset default)')

    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU device ID(s) to use (e.g., "0", "1", "0,1"). If not specified, uses all available GPUs.')

    args = parser.parse_args()

    # Resolve dataset-specific defaults
    ds_cfg = DATASET_CONFIGS[args.dataset]
    dataset_path = args.dataset_path or ds_cfg['dataset_path']
    output_dir = args.output_dir or ds_cfg['output_dir']
    denoised_cache_dir = args.denoised_cache_dir or ds_cfg['denoised_cache_dir']
    dataset_suffix_prefix = ds_cfg['file_suffix_prefix']
    dataset_label = ds_cfg['dataset_label']

    # Validate model selection
    if not validate_model_selection(args.models):
        return

    selected_models = expand_model_selection(args.models)

    print(f"Dataset: {dataset_label}")
    print(f"Selected models for processing: {selected_models}")
    print(f"Total models selected: {len(selected_models)}")

    # Set random seed
    set_random_seed(args.random_seed)

    # Configure GPU devices
    configure_gpu(args.gpu_id)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'training_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Load dataset
    print(f"\n{'='*60}")
    print(f"LOADING DATASET ({dataset_label})")
    print(f"{'='*60}")
    print(f"Loading dataset from {dataset_path}...")
    start_time = time.time()
    dataset = load_radioml_data(dataset_path)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    # Explore dataset
    if args.mode in ['explore', 'all']:
        print(f"\n{'='*60}")
        print("EXPLORING DATASET")
        print(f"{'='*60}")
        mods, snrs = explore_dataset(dataset)
        plot_signal_examples(dataset, mods, os.path.join(output_dir, 'exploration'))

    # Prepare data
    if args.mode in ['train', 'evaluate', 'all']:
        print(f"\n{'='*60}")
        print("PREPARING DATA")
        print(f"{'='*60}")

        print("Using stratified splitting by (modulation type, SNR) combinations...")
        X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_with_cache_suffix(
            dataset,
            cache_filename_suffix=dataset_suffix_prefix,
            augment_data=args.augment_data,
            denoising_method=args.denoising_method,
            denoised_cache_dir=denoised_cache_dir
        )

    # Training
    if args.mode in ['train', 'all']:
        print(f"\n{'='*60}")
        print("TRAINING SELECTED MODELS")
        print(f"{'='*60}")

        suffix = get_file_suffix(args.denoising_method, args.augment_data, dataset_suffix_prefix)
        suffix += "_stratified"

        input_shape = X_train.shape[1:]
        num_classes = len(mods)

        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Models to train: {selected_models}")

        train_selected_models(
            selected_models, X_train, y_train, X_val, y_val,
            input_shape, num_classes, models_dir, plots_dir,
            suffix, args.batch_size, args.epochs
        )

    # Evaluation
    if args.mode in ['evaluate', 'all']:
        print(f"\n{'='*60}")
        print("EVALUATING SELECTED MODELS")
        print(f"{'='*60}")

        suffix = get_file_suffix(args.denoising_method, args.augment_data, dataset_suffix_prefix)
        suffix += "_stratified"

        print(f"Models to evaluate: {selected_models}")

        evaluate_selected_models(
            selected_models, X_test, y_test, snr_test, mods,
            models_dir, results_dir, suffix
        )

    print(f"\n{'='*60}")
    print("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_label}")
    print(f"Processed models: {selected_models}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

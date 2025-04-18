import argparse
import os
from typing import Any, Dict, Optional


def str2bool(v: Any) -> bool:
    """Convert string to boolean value.
    
    Args:
        v: Value to convert to boolean
        
    Returns:
        bool: Converted boolean value
        
    Raises:
        argparse.ArgumentTypeError: If value cannot be converted to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset-related arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset', default='lrw', help='Dataset selection')
    dataset_group.add_argument('--num-classes', type=int, default=500, help='Number of classes (LRW)')
    dataset_group.add_argument('--modality', default='video', 
                             choices=['video', 'raw_audio'], 
                             help='Input modality type')


def add_directory_args(parser: argparse.ArgumentParser) -> None:
    """Add directory-related arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    dir_group = parser.add_argument_group('Directory Configuration')
    dir_group.add_argument('--data-dir', default='/home/compu/sws/lipreading/datasets/visual_data', 
                          help='Directory containing input data')
    dir_group.add_argument('--label-path', type=str, 
                          default='/home/compu/sws/lipreading/datasets/labels/500WordsSortedList.txt', 
                          help='Path to label file')
    dir_group.add_argument('--annonation-direc', default='/home/compu/sws/lipreading/datasets/info_txt', 
                          help='Directory containing annotation files')


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-related arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    model_group = parser.add_argument_group('Model Configuration')
    
    # Embedding model args
    emb_group = model_group.add_argument_group('Embedding Model')
    emb_group.add_argument('--emb-size', type=int, default=512, 
                          help='Embedding dimension size')
    emb_group.add_argument('--backbone-type', type=str, default='resnet',
                          choices=['resnet', 'efficientnet_v2_t_inc_24',
                                 'efficientnet_v2_t_inc_32', 'efficientnet_v2_t_inc_64'],
                          help='Backbone architecture')
    emb_group.add_argument('--relu-type', type=str, default='prelu',
                          choices=['silu', 'prelu'],
                          help='Activation function type')
    
    # Backend model args
    backend_group = model_group.add_argument_group('Backend Model')
    backend_group.add_argument('--config-path', type=str,
                             default='./td3net_configs/td3net_config_base.yaml',
                             help='Path to model configuration file')
    
    # Model variant args
    variant_group = model_group.add_argument_group('Model Variants')
    variant_group.add_argument('--use-td3-block', type=str2bool, default=True,
                             help='Use TD3 block (True: TD3Net/TD2Net, False: Dense-TCN)')
    variant_group.add_argument('--use-multi-dilation', type=str2bool, default=True,
                             help='Use multi-dilation (True: TD3Net/TD2Net, False: Dense-TCN)')
    variant_group.add_argument('--use-bottle-layer', type=str2bool, default=True,
                             help='Use bottleneck layer in blocks')


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training-related arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--action', type=str, default='train',
                           help='Mode: train or test')
    train_group.add_argument('--resume-id', type=str, default=None,
                           help='Neptune resume ID')
    train_group.add_argument('--model-type', default='td3net',
                           help='Model type (tcn or td3net)')
    train_group.add_argument('--batch-size', type=int, default=32,
                           help='Mini-batch size')
    train_group.add_argument('--optimizer', type=str, default='adamw',
                           choices=['adam', 'sgd', 'adamw'],
                           help='Optimizer type')
    train_group.add_argument('--lr', default=3e-4, type=float,
                           help='Initial learning rate')
    train_group.add_argument('--init-epoch', default=0, type=int,
                           help='Starting epoch number')
    train_group.add_argument('--epochs', default=120, type=int,
                           help='Total number of epochs')
    train_group.add_argument('--amp', type=str2bool, default=True,
                           help='Use automatic mixed precision')
    
    # Augmentation args
    aug_group = train_group.add_argument_group('Augmentation')
    aug_group.add_argument('--alpha', default=0.4, type=float,
                          help='Mixup interpolation strength (uniform=1., ERM=0.)')
    aug_group.add_argument('--var-length', type=str2bool, default=False,
                          help='Use variable length augmentation')


def add_testing_args(parser: argparse.ArgumentParser) -> None:
    """Add testing-related arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    test_group = parser.add_argument_group('Testing Configuration')
    test_group.add_argument('--model-path', type=str, default=None,
                          help='Path to pretrained model')
    test_group.add_argument('--allow-size-mismatch', type=str2bool, default=False,
                          help='Allow initialization from model with mismatching weight tensors')


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add logging-related arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    log_group = parser.add_argument_group('Logging Configuration')
    log_group.add_argument('--logging-dir', type=str, default='./train_logs',
                         help='Directory to save log files')
    log_group.add_argument('--cal_model_size', type=str2bool, default=False,
                         help='Calculate model size')
    log_group.add_argument('--ex-name', default='temp',
                         help='Experiment name')
    log_group.add_argument('--neptune_logging', type=str2bool, default=False,
                         help='Enable Neptune logging')


def add_misc_args(parser: argparse.ArgumentParser) -> None:
    """Add miscellaneous arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    misc_group = parser.add_argument_group('Miscellaneous Configuration')
    misc_group.add_argument('--workers', default=8, type=int,
                          help='Number of data loading workers')


def load_args(default_config: Optional[Dict] = None) -> argparse.Namespace:
    """Load and parse command line arguments.
    
    Args:
        default_config: Optional dictionary of default configurations
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch Lipreading')
    
    # Add all argument groups
    add_dataset_args(parser)
    add_directory_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_testing_args(parser)
    add_logging_args(parser)
    add_misc_args(parser)
    
    args = parser.parse_args()
    return args
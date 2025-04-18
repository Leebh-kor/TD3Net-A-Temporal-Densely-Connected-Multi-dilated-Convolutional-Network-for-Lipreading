import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .model import TD3Net_Lipreading

# Load environment variables
load_dotenv()

class ModelConfig:
    """Configuration loader for TD3Net model."""
    
    @staticmethod
    def load_td3net_config(args: Any, config_path: str) -> None:
        """Load TD3Net specific configurations from the config file.
        
        Args:
            args: Arguments object to update
            config_path: Path to the config file
        """
        args_loaded = OmegaConf.load(config_path)
        config_keys = [
            'growth_rate', 'num_layers', 'num_td2_blocks', 'out_block',
            'block_comp', 'trans_comp_factor', 'kernel_size', 'is_cbr', 'dropout_p'
        ]
        for key in config_keys:
            setattr(args, key, args_loaded[key])

    @staticmethod
    def get_model_options(args: Any) -> Dict[str, Any]:
        """Create model options dictionary.
        
        Args:
            args: Arguments containing model parameters
            
        Returns:
            Dictionary of model options
        """
        return {
            'emb_size': args.emb_size,
            'backbone_type': args.backbone_type,
            'relu_type': args.relu_type,
            'growth_rate': args.growth_rate,
            'num_layers': args.num_layers,
            'num_td2_blocks': args.num_td2_blocks,
            'out_block': args.out_block,
            'block_comp': args.block_comp,
            'trans_comp_factor': args.trans_comp_factor,
            'kernel_size': args.kernel_size,
            'is_cbr': args.is_cbr,
            'dropout_p': args.dropout_p,
            'use_td3_block': args.use_td3_block,
            'use_multi_dilation': args.use_multi_dilation,
            'use_bottle_layer': args.use_bottle_layer
        }

def get_model_from_configs(args: Any) -> nn.Module:
    """Create and return a model based on the provided configuration.
    
    Args:
        args: Configuration object containing model parameters
        
    Returns:
        CUDA-enabled model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    if args.model_type == 'td3net':
        ModelConfig.load_td3net_config(args, args.config_path)
        model_options = ModelConfig.get_model_options(args)
        model = TD3Net_Lipreading(**model_options)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    return model.cuda()

class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def calculate_norm2(model: nn.Module) -> None:
        """Calculate and print the 2-norm of the model parameters."""
        para_norm = sum(p.data.norm(2) for p in model.parameters())
        print(f'2-norm of the neural network: {para_norm**.5:.4f}')

    @staticmethod
    def show_lr(optimizer: Optimizer) -> float:
        """Get the current learning rate."""
        return optimizer.param_groups[0]['lr']

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FileIO:
    """File input/output operations."""
    
    @staticmethod
    def read_txt_lines(filepath: str) -> List[str]:
        """Read lines from a text file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath) as myfile:
            return myfile.read().splitlines()

    @staticmethod
    def save_as_json(data: Dict, filepath: str) -> None:
        """Save dictionary as JSON file."""
        with open(filepath, 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)

    @staticmethod
    def load_json(json_fp: str) -> Dict:
        """Load JSON file."""
        if not os.path.isfile(json_fp):
            raise FileNotFoundError(f"JSON file not found: {json_fp}")
        with open(json_fp, 'r') as f:
            return json.load(f)

    @staticmethod
    def save2npz(filename: str, data: np.ndarray) -> None:
        """Save numpy array to npz file."""
        if data is None:
            raise ValueError("Data cannot be None")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(filename, data=data)

class CheckpointSaver:
    """Handles model checkpoint saving and loading."""
    
    def __init__(self, save_dir: str, mode: str = 'max',
                 checkpoint_fn: str = 'ckpt.pth.tar',
                 best_fn: str = 'ckpt.best.pth.tar'):
        self.save_dir = save_dir
        self.mode = mode
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.current_best = -1 if mode == 'max' else np.inf

    def save(self, save_dict: Dict, current_perf: float, epoch: int = -1) -> None:
        """Save checkpoint and keep copy if current performance is best overall."""
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)
        
        self.is_best = (current_perf > self.current_best) if self.mode == 'max' else (current_perf < self.current_best)
            
        if self.is_best:
            self.current_best = current_perf
            best_fp = os.path.join(self.save_dir, self.best_fn)
            
        save_dict['best_prec'] = self.current_best
        torch.save(save_dict, checkpoint_fp)
        print(f"Checkpoint saved at {checkpoint_fp}")
        
        if self.is_best:
            shutil.copyfile(checkpoint_fp, best_fp)

    def set_best_from_ckpt(self, ckpt_dict: Dict) -> None:
        """Set best performance from checkpoint dictionary."""
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage', None)

class ModelLoader:
    """Handles model loading operations."""
    
    @staticmethod
    def load_model(load_path: str, model: nn.Module,
                  optimizer: Optional[Optimizer] = None,
                  scheduler: Optional[_LRScheduler] = None,
                  allow_different_module_names: bool = False,
                  allow_size_mismatch: bool = False) -> Union[nn.Module, Tuple]:
        """Load model from file."""
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        checkpoint = torch.load(load_path)
        loaded_state_dict = checkpoint['model_state_dict']

        if allow_size_mismatch:
            ModelLoader._handle_size_mismatch(loaded_state_dict, model)
            
        if allow_different_module_names:
            loaded_state_dict = ModelLoader._handle_different_module_names(loaded_state_dict, model)
            
        model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
        
        if optimizer is not None and scheduler is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return model, optimizer, scheduler, checkpoint['scheduler_state_dict']['last_epoch'], checkpoint

        return model

    @staticmethod
    def _handle_size_mismatch(loaded_state_dict: Dict, model: nn.Module) -> None:
        """Handle parameter size mismatches between loaded and current model."""
        loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
        model_sizes = {k: v.shape for k, v in model.state_dict().items()}
        
        for k in list(loaded_sizes.keys()):
            if loaded_sizes[k] != model_sizes[k]:
                del loaded_state_dict[k]

    @staticmethod
    def _handle_different_module_names(loaded_state_dict: Dict, model: nn.Module) -> Dict:
        """Handle different module names between loaded and current model."""
        pretrained_items = list(loaded_state_dict.items())
        model_state_dict = model.state_dict()
        
        for (k, v), (layer_name, weights) in zip(model_state_dict.items(), pretrained_items):
            model_state_dict[k] = weights
            
        return model_state_dict

class Logger:
    """Handles logging operations."""
    
    @staticmethod
    def get_logger(save_path: str) -> logging.Logger:
        """Initialize and return a logger."""
        log_path = f'{save_path}_log.txt'
        logger = logging.getLogger("mylog")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        
        return logger

    @staticmethod
    def update_logger_batch(args: Any, logger: logging.Logger, dset_loader: torch.utils.data.DataLoader,
                          batch_idx: int, running_loss: float, running_corrects: float,
                          running_all: int, batch_time: AverageMeter,
                          data_time: AverageMeter, inter_time: AverageMeter) -> None:
        """Update logger with batch information."""
        perc_epoch = 100. * batch_idx / (len(dset_loader)-1)
        logger.info(
            '[{:5.0f}/{:5.0f} ({:.0f}%)] Loss: {:.4f} Acc:{:.4f} Cost time:{:1.3f} ({:1.3f})s'.format(
                running_all, len(dset_loader.dataset), perc_epoch,
                running_loss / running_all, running_corrects / running_all,
                batch_time.sum, inter_time.sum
            )
        )
        batch_time.reset()

class PathUtils:
    """Path-related utility functions."""
    
    @staticmethod
    def get_save_folder(logging_dir: str, ex_name: str) -> str:
        """Create and return save folder path."""
        save_path = Path(logging_dir) / ex_name
        save_path.mkdir(parents=True, exist_ok=True)
        return str(save_path)

def seed_init(seed: int = int(os.getenv('DEFAULT_SEED', 325))) -> None:
    """Initialize random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class NeptuneLogger:
    """Handles Neptune.ai logging operations."""
    
    @staticmethod
    def init_neptune(params: Dict, resume_id: Optional[str] = None) -> neptune.Run:
        """Initialize Neptune logger."""
        project = os.getenv('NEPTUNE_PROJECT')
        api_token = os.getenv('NEPTUNE_API_TOKEN')
        
        if not all([project, api_token]):
            raise ValueError("Neptune project and API token must be set in .env file")
            
        if resume_id:
            logger = neptune.init_run(
                project=project,
                api_token=api_token,
                with_id=resume_id
            )
            print(f'\nExperiment {resume_id} logger resumed.')
        else:
            logger = neptune.init_run(
                project=project,
                api_token=api_token
            )
            logger['parameters'] = params

        return logger

def get_params(args: Any) -> Dict:
    """Extract parameters from arguments object."""
    params = {}
    args_dict = vars(args)
    
    for key, value in args_dict.items():
        if not key.startswith('__'):
            if isinstance(value, dict):
                params.update(value)
            else:
                params[key] = value
                
    return params


import os
import warnings

from config import load_args
from lipreading.dataloaders import get_data_loaders
from lipreading.eval import Tester
from lipreading.train import Trainer
from lipreading.utils import (
    CheckpointSaver,
    Logger,
    NeptuneLogger,
    PathUtils,
    get_params,
    seed_init,
)

warnings.filterwarnings(action='ignore')

def setup_logging(args):
    """Initialize logging and checkpoint saving."""
    save_path = PathUtils.get_save_folder(args.logging_dir, args.ex_name)
    ckpt_saver = CheckpointSaver(save_path, mode='max')
    logger_save_path = os.path.join(save_path, args.config_path.split('/')[-1])
    logger = Logger.get_logger(logger_save_path)
    return save_path, ckpt_saver, logger

def setup_neptune(args, logger):
    """Initialize Neptune logging if enabled."""
    if args.neptune_logging:
        logger.info('---Do neptune logging---')
        neptune_logger = NeptuneLogger.init_neptune(get_params(args), args.resume_id)
    else:
        logger.info('---Do not neptune logging---')
        neptune_logger = None
    return neptune_logger

def run_training(args, dset_loaders, ckpt_saver, logger, neptune_logger):
    """Execute training and evaluation pipeline."""
    trainer = Trainer(args, dset_loaders, ckpt_saver, logger, neptune_logger)
    trainer.train()
    
    tester = Tester(args, dset_loaders, logger, neptune_logger)
    tester.evaluate()

def run_testing(args, dset_loaders, neptune_logger):
    """Execute testing pipeline."""
    tester = Tester(args, dset_loaders, neptune_logger=neptune_logger)
    tester.evaluate()

def main():
    # Load and print configuration
    args = load_args()
    print(vars(args))
    seed_init()
    
    # Setup logging and Neptune
    save_path, ckpt_saver, logger = setup_logging(args)
    neptune_logger = setup_neptune(args, logger)
    
    try:
        # Get data loaders based on action
        if args.action == 'train':
            dset_loaders = get_data_loaders(args, ['train', 'test'])
            run_training(args, dset_loaders, ckpt_saver, logger, neptune_logger)
        else:
            dset_loaders = get_data_loaders(args, ['test'])
            run_testing(args, dset_loaders, neptune_logger)
            
    finally:
        # Ensure Neptune is properly stopped
        if args.neptune_logging and neptune_logger:
            neptune_logger.stop()

if __name__ == "__main__":
    main() 
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import Logger, ModelConfig, ModelLoader, get_model_from_configs


class Tester:
    def __init__(self, args, dset_loader, logger=None, neptune_logger=None):
        self.args = args
        self.dset_loader = dset_loader['test']
        self.criterion = nn.CrossEntropyLoss()
        self.model = get_model_from_configs(self.args)
        best_fp = os.path.join('{}/{}'.format(self.args.logging_dir, self.args.ex_name), 'ckpt.best.pth.tar') if not self.args.model_path else self.args.model_path
        self.model = ModelLoader.load_model(
            best_fp, 
            self.model, 
            allow_different_module_names=True
        )
        self.logger = logger
        self.neptune_logger = neptune_logger
        
    def evaluate(self):
        self.model.eval()

        running_loss = 0.
        running_corrects = 0.

        with torch.no_grad():
            for batch_idx, (input, lengths, labels) in enumerate(tqdm(self.dset_loader)):
                logits = self.model(input.unsqueeze(1).cuda(), lengths=lengths).cpu().detach()
                _, preds = torch.max(F.softmax(logits, dim=1), dim=1)
                running_corrects += preds.eq(labels.view_as(preds)).sum().item()
                loss = self.criterion(logits, labels)
                running_loss += loss.item() * input.size(0)
                
        test_loss = running_loss / len(self.dset_loader.dataset)
        test_acc = running_corrects / len(self.dset_loader.dataset)
        
        if self.args.neptune_logging:
            self.neptune_logger['logs/test loss'].log(test_loss)
            self.neptune_logger['logs/test acc'].log(test_acc)
            
        if self.logger is None:
            print('---Test score---')
            print(f'Test-time performance : Loss: {test_loss:.4f}\t Acc:{test_acc:.4f}')
        else:
            self.logger.info('---Test score---')
            self.logger.info(f'Test-time performance : Loss: {test_loss:.4f}\t Acc:{test_acc:.4f}')
                
        
            
        
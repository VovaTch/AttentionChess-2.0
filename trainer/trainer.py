import copy

import numpy as np
import torch
from torchvision.utils import make_grid
from colorama import Fore

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.self_play_flag = True if 'mcts' in config else False

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        if self.config['lr_scheduler']['type'] == 'OneCycleLR':
            oclr_scheduler = copy.deepcopy(self.lr_scheduler)
        self.logger.debug(Fore.YELLOW + '\n -------------- << T R A I N I N G >> --------------\n' + Fore.RESET)
        
        for batch_idx, target_dict in enumerate(self.data_loader):
            
            for target in target_dict:
                target_dict[target] = target_dict[target].to(self.device)

            self.optimizer.zero_grad()
            data = target_dict['board']
            output = self.model.forward_raw(data)
            loss_dict = self.criterion(output, target_dict)
            
            loss = sum([loss_dict[loss_type] * self.config['loss_weights'][loss_type]
                        for loss_type in self.config['loss_weights']])
            loss += sum([loss_dict[loss_type] * self.config['loss_weights'][loss_type[:-6]]
                        for loss_type in loss_dict.keys() if loss_type[:-6] in self.config['loss_weights']])
            
            loss.backward()
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(self.criterion, output, target_dict))

            if batch_idx % self.log_step == 0:
                self.logger.debug(Fore.GREEN + f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss:' + 
                                  Fore.CYAN + f' {loss.item():.6f}' + Fore.RESET)

            if batch_idx == self.len_epoch:
                break
            
            if self.config['lr_scheduler']['type'] == 'OneCycleLR':
                oclr_scheduler.step()
                
            # Set models in case of full self play; needed because there are 2 MCTS objects
            if self.self_play_flag:
                self.data_loader.dataset.mcts.good_model = copy.deepcopy(self.model)
                self.data_loader.dataset.mcts.evil_model = copy.deepcopy(self.model)
            
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{Fore.LIGHTMAGENTA_EX + 'val_'+ k + Fore.RESET : v for k, v in val_log.items()})

        if self.lr_scheduler is not None and self.config['lr_scheduler']['type'] != 'OneCycleLR':
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        self.logger.debug(Fore.LIGHTRED_EX + '\n -------------- << E V A L U A T I O N >> --------------\n' + Fore.RESET)
        
        with torch.no_grad():
            for batch_idx, target_dict in enumerate(self.data_loader):
            
                for target in target_dict:
                    target_dict[target] = target_dict[target].to(self.device)

                self.optimizer.zero_grad()
                data = target_dict['board']
                output = self.model.forward_raw(data)
                loss_dict = self.criterion(output, target_dict)
                
                loss = sum([loss_dict[loss_type] * self.config['loss_weights'][loss_type]
                            for loss_type in self.config['loss_weights']])
                loss += sum([loss_dict[loss_type] * self.config['loss_weights'][loss_type[:-6]]
                            for loss_type in loss_dict.keys() if loss_type[:-6] in self.config['loss_weights']])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(self.criterion, output, target_dict))

                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

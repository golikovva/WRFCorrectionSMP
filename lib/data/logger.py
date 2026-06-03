import logging
import os
import re

import numpy as np
import torch
import pickle


class WRFLogger:
    def __init__(self, cfg, base_log_dir=None, folder_name=None, save_dir=None):
        if save_dir is not None:
            self.save_dir = os.fspath(save_dir)
            os.makedirs(self.save_dir, exist_ok=True)
            self.folder_path = os.path.dirname(self.save_dir)
            self.experiment_number = os.path.basename(self.save_dir)
        else:
            if base_log_dir is None:
                base_log_dir = '/home/logs'
            if folder_name is None:
                folder_name = 'unknown'

            if not os.path.exists(os.path.join(base_log_dir, folder_name)):
                os.makedirs(os.path.join(base_log_dir, folder_name))
            self.folder_path = os.path.join(base_log_dir, folder_name)

            if cfg.run_config.run_mode == 'test':
                self.experiment_number = cfg.test_config.run_id
            else:
                self.experiment_number = self.get_experiment_number()

            self.save_dir = os.path.join(self.folder_path, f'misc_{self.experiment_number}')

        self.model_save_dir = os.path.join(self.save_dir, 'models')
        self.log_dir = os.path.join(self.save_dir, 'logs')
        self.plots_dir = os.path.join(self.save_dir, 'plots')

        os.makedirs(self.log_dir) if not os.path.exists(self.log_dir) else None
        os.makedirs(self.model_save_dir) if not os.path.exists(self.model_save_dir) else None
        os.makedirs(self.plots_dir) if not os.path.exists(self.plots_dir) else None

        self.logger = self.create_logger()
        self.logger.info(f"Testing the custom logger for module {__name__}...")

        self.train_loss = []
        self.loss_evolution = []
        self.best_epoch = -1
        self.mse = 0
        self.mse1 = 0
        self.mse2 = 0
        self.mse3 = 0
        self.mse4 = 0
        self.iters_counted = 0
        self.betas = [1]

    def create_logger(self):
        logger_name = f"{__name__}.{abs(hash(os.path.abspath(self.save_dir)))}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        handler = logging.FileHandler(f"{os.path.join(self.log_dir, __name__)}.log", mode='a')
        formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    def get_experiment_number(self):
        numbers = set()
        for directory in os.listdir(self.folder_path):
            if re.match(r'(misc)_\d+$', directory):
                numbers.add(int(directory.split('_')[-1]))
        return max(numbers) + 1 if len(numbers) else 1

    def set_beta(self, betas):
        self.betas = betas

    def accumulate_stat(self, mse, mse1=None, mse2=None, mse3=None, mse4=None):  # todo покрасивее работать с бетами
        self.mse += float(mse)
        if mse1:
            self.mse1 += float(mse1)
        if mse2:
            self.mse2 += float(mse2)
        if mse3:
            self.mse3 += float(mse3)
        if mse4:
            self.mse4 += float(mse4)
        self.iters_counted += 1

    def reset_stat(self):
        self.mse = 0
        self.mse1 = 0
        self.mse2 = 0
        self.mse3 = 0
        self.mse4 = 0
        self.iters_counted = 0

    def print_stat_readable(self, epoch=None, reset=True):
        betas = self.betas if self.betas is not None else 'beta'

        if epoch:
            self.logger.info(f"Validation epoch {epoch} successful with val loss:")
        else:
            self.logger.info(f"Validation successful with val loss:")
        if self.mse > 0:
            mse = round(self.mse / self.iters_counted, 5)
            self.logger.info(f"    MSE + deltaMSE: {mse}")
            self.loss_evolution.append(mse)
        if self.mse2 > 0 or self.mse3 > 0 or self.mse4 > 0:
            self.logger.info(f"    MSE: {round(self.mse1 / self.iters_counted, 5)}"
                             f" + {betas[1]} * deltaMSE: {round(self.mse2 / self.iters_counted, 5)}"
                             f" + {betas[2]} * stationMSE: {round(self.mse3 / self.iters_counted, 5)}"
                             f" + {betas[3]} * scatterMSE: {round(self.mse4 / self.iters_counted, 5)}")
        if reset:
            self.reset_stat()

    def get_stat(self):
        mse0 = self.mse / self.iters_counted
        mse1 = self.mse1 / self.iters_counted
        mse2 = self.mse2 / self.iters_counted
        mse3 = self.mse3 / self.iters_counted
        mse4 = self.mse4 / self.iters_counted
        return mse0, mse1, mse2, mse3, mse4

    def save_model(self, model_state_dict, epoch):
        loss = self.loss_evolution
        if len(loss) > 0 and loss.index(min(loss)) == len(loss) - 1:
            torch.save(model_state_dict, os.path.join(self.model_save_dir, f'model_{epoch}.pth'))
            old_model_path = os.path.join(self.model_save_dir, f'model_{self.best_epoch}.pth')
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
            self.best_epoch = len(loss) - 1
        np.save(os.path.join(self.log_dir, 'val_loss'), np.stack(self.loss_evolution))
        np.save(os.path.join(self.log_dir, 'train_loss'), np.stack(self.train_loss))
        return self.best_epoch

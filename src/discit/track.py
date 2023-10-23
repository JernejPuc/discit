"""Tracking"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer


class CheckpointTracker:
    meta: 'dict[str, Any]'
    rng: np.random.Generator
    logger: logging.Logger

    def __init__(
        self,
        model_name: str = 'model',
        data_dir: str = 'data',
        device: str = 'cuda',
        initial_seed: int = 42,
        transfer_name: str = '',
        ver_to_transfer: int = None,
        reset_step_on_transfer: bool = False,
        deterministic: bool = False
    ):
        if deterministic:
            # See: https://github.com/pytorch/pytorch/issues/76176
            print('Warning: Some algorithms have unresolved issues in deterministic mode.')

            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        self.model_name = model_name
        self.data_dir = os.path.join(data_dir, model_name)
        self.device = device

        self.log_path = os.path.join(self.data_dir, 'log.txt')
        self.resume(initial_seed, transfer_name, ver_to_transfer, reset_step_on_transfer)

        self.model: 'Module | None' = None
        self.optimiser: 'Optimizer | None' = None

    def resume(self, seed: int, transfer_name: str, ver_to_transfer: int, reset_step_on_transfer: bool):
        """Initialise the first or load the last checkpoint data of the current model."""

        # Load or create meta.json file
        if os.path.exists(path := os.path.join(self.data_dir, 'meta.json')):
            with open(path, 'r') as meta_file:
                meta_data: 'dict[str, Any]' = json.load(meta_file)

        else:
            os.makedirs(self.data_dir, exist_ok=True)

            with open(path, 'w') as meta_file:
                json.dump({}, meta_file)

            # Start from existing files
            if transfer_name and os.path.exists(
                path := os.path.join(os.path.dirname(self.data_dir), transfer_name, 'meta.json')
            ):
                with open(path, 'r') as meta_file:
                    meta_data: 'dict[str, Any]' = json.load(meta_file)

            else:
                meta_data = None

        # Init. logger
        log_handler = logging.FileHandler(self.log_path)
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        self.logger = logging.getLogger(name=self.model_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(log_handler)

        # Load or create initial meta data
        if meta_data:
            self.meta = meta_data[sorted(meta_data.keys())[-1]]

            if transfer_name:
                self.logger.info(f'Starting from model {transfer_name} ver. {ver_to_transfer}.')

                self.meta['name'] = self.model_name

                if ver_to_transfer is not None:
                    transfer_dir = os.path.join(os.path.dirname(self.data_dir), transfer_name)

                    self.meta['ckpt_ver'] = ver_to_transfer
                    self.meta['model_path'] = os.path.join(transfer_dir, f'model_{ver_to_transfer:03d}.pt')
                    self.meta['ckpt_path'] = os.path.join(transfer_dir, f'ckpt_{ver_to_transfer:03d}.pt')

                if reset_step_on_transfer:
                    self.meta['epoch_step'] = 0
                    self.meta['update_step'] = 0

        else:
            self.meta = {'ckpt_ctr': -1, 'ckpt_ver': 0, 'name': self.model_name}
            self.update()

        # Load checkpoint data and set RNG states
        if os.path.exists(path := self.meta['ckpt_path']):
            ckpt = torch.load(path, map_location=self.device)

            self.rng = np.random.default_rng()
            self.rng.__setstate__(ckpt['np_rng'])

            torch.set_rng_state(torch.tensor(ckpt['pt_rng'], dtype=torch.uint8))

            if self.device.startswith('cuda') and ckpt['pt_rng_cuda']:
                torch.cuda.set_rng_state(torch.tensor(ckpt['pt_rng_cuda'], dtype=torch.uint8), self.device)

            log_text = f'Resumed state from ckpt. {self.meta["ckpt_ver"]}.'

        # Init. new RNG states
        else:
            if seed is None:
                seed = torch.initial_seed()

            self.rng = np.random.default_rng(seed)
            torch.random.manual_seed(seed)

            log_text = f'Creating state ckpt. {self.meta["ckpt_ver"]}.'

        # Report on init or load event via log.txt file
        self.logger.info(log_text)

        print(log_text)

    def update(self, epoch_step: int = 0, update_step: int = 0, ckpt_increment: int = 0, score: float = None):
        self.meta['ckpt_ctr'] += 1
        self.meta['ckpt_ver'] += ckpt_increment
        self.meta['epoch_step'] = epoch_step
        self.meta['update_step'] = update_step
        self.meta['perf_score'] = score
        self.meta['model_path'] = os.path.join(self.data_dir, f'model_{self.meta["ckpt_ver"]:03d}.pt')
        self.meta['ckpt_path'] = os.path.join(self.data_dir, f'ckpt_{self.meta["ckpt_ver"]:03d}.pt')

    def checkpoint(self, epoch_step: int = 0, update_step: int = 0, ckpt_increment: int = 0, score: float = None):
        """Save current data."""

        # Update version and paths
        self.update(epoch_step, update_step, ckpt_increment, score)

        # Add to meta map
        with open(os.path.join(self.data_dir, 'meta.json'), 'r') as meta_file:
            meta_data = json.load(meta_file)

        meta_data[datetime.now().strftime('%Y-%m-%d %H:%M:%S')] = self.meta

        # Save meta backup
        with open(os.path.join(self.data_dir, 'meta_backup.json'), 'w') as meta_file:
            json.dump(meta_data, meta_file)

        # Save meta proper
        with open(os.path.join(self.data_dir, 'meta.json'), 'w') as meta_file:
            json.dump(meta_data, meta_file)

        # Save model params.
        model_state = None if self.model is None else self.model.state_dict()

        if model_state is not None:
            torch.save(model_state, self.meta['model_path'])

        # Save full checkpoint
        optim_state = None if self.optimiser is None else self.optimiser.state_dict()

        torch.save(
            {
                'model': model_state,
                'optim': optim_state,
                'np_rng': self.rng.__getstate__(),
                'pt_rng': torch.get_rng_state().tolist(),
                'pt_rng_cuda': torch.cuda.get_rng_state(self.device).tolist() if self.device.startswith('cuda') else [],
                **self.meta},
            self.meta['ckpt_path'])

        # Report on checkpoint event via log
        log_text = f'Saved ckpt. ver. {self.meta["ckpt_ver"]} on epoch {self.meta["epoch_step"]}.'

        self.logger.info(log_text)

        print(f'\n{log_text}')

    def load_model(self, model: Module, optimiser: Optimizer = None):
        """Restore model and optimiser params."""

        self.model = model.to(self.device)
        self.optimiser = optimiser

        if path_exists := os.path.exists(path := self.meta['ckpt_path']):
            ckpt = torch.load(path, map_location=self.device)

            if (state := ckpt['model']) is not None:
                model.load_state_dict(state)

            if optimiser is not None and (state := ckpt['optim']) is not None:
                optimiser.load_state_dict(state)

        print(f'{"Loaded" if path_exists else "Initialised"} model ver. {self.meta["ckpt_ver"]}.')

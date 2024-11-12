"""Multi-agent reinforcement learning"""

import os
from abc import ABC, abstractmethod
from datetime import timedelta
from io import BytesIO
from time import perf_counter
from typing import Callable

import torch
from torch import cuda, Tensor
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from .accel import capture_graph
from .data import ExperienceBuffer, TensorDict
from .distr import Distribution
from .optim import MultiOptimizer, MultiScheduler
from .track import CheckpointTracker


class MultiActorCritic(Module, ABC):
    def __init__(self):
        Module.__init__(self)

    def init_mem(self, n_actors: int = None, n_envs: int = None) -> 'tuple[Tensor, ...]':
        return ()

    def reset_mem(self, mem: 'tuple[Tensor, ...]', nonreset_mask: Tensor) -> 'tuple[Tensor, ...]':
        return ()

    @abstractmethod
    def get_distr(self, args: 'tuple[Tensor, ...]') -> Distribution:
        ...

    def unwrap_sample(self, act: 'tuple[Tensor, ...]', aux: 'tuple[Tensor, ...]') -> 'tuple[Tensor | None, ...]':
        return act[0], None

    @abstractmethod
    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        act: 'tuple[Tensor, ...] | None'
    ) -> 'tuple[dict[str, Tensor | tuple[Tensor, ...]], tuple[Tensor, ...], tuple[Tensor, ...]]':
        ...

    @abstractmethod
    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        act: 'tuple[Tensor, ...]',
        detach: bool = True
    ) -> 'dict[str, Distribution | tuple[Distribution, ...] | Tensor | tuple[Tensor, ...]]':
        ...


class AuxTask(ABC):
    STAT_KEYS = ()

    def __init__(self, online: bool = False, offline: bool = False):
        self.online = online
        self.offline = offline
        self.n_update_steps = 0

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def collect(self, batch: TensorDict, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):
        ...

    @abstractmethod
    def update(self, batches: 'list[TensorDict]', stats: 'dict[str, Tensor]'):
        ...

    @abstractmethod
    def loss(
        self,
        batch: TensorDict,
        act: Distribution,
        vals: 'tuple[Distribution, ...]',
        auxs: 'tuple[Distribution, ...]',
        stats: 'dict[str, Tensor]'
    ) -> Tensor:
        ...


class MAXPPO:
    """
    Multi-agent variant of PPO with joint targets and external
    (e.g. mixed or distilled) individual advantage estimates.
    """

    MAX_DISP_SECONDS = 99*24*3600

    STAT_KEYS = (
        'Out/act_mean',
        'Out/act_std',
        'Out/val_mean',
        'Env/reward',
        'Env/resets',
        'Main/loss',
        'Main/policy',
        'Main/value',
        'Val/adv',
        'Val/tot',
        'Val/ind',
        'Aux/loss',
        'Main/entropy',
        'Main/ratio_diff',
        'GAE/adv_mean',
        'GAE/adv_std')

    def __init__(
        self,
        env_step: Callable[
            ['Tensor | None', 'Tensor | None'],
            'tuple[tuple[Tensor, ...], dict[str, Tensor | tuple[Tensor, ...]], dict[str, float]]'],
        ckpt_tracker: CheckpointTracker,
        scheduler: MultiScheduler,
        n_envs: int,
        n_actors: int,
        n_epochs: int,
        log_epoch_interval: int = 1,
        ckpt_epoch_interval: int = 1,
        branch_epoch_interval: int = 0,
        n_rollout_steps: int = 1,
        n_truncated_steps: int = 1,
        n_passes_per_step: int = 1,
        buffer_size: int = None,
        batch_size: int = None,
        discount_gammas: 'float | tuple[float, ...]' = 0.99,
        trace_lambda: float = 0.95,
        clip_ratio: float = 0.25,
        policy_weight: float = 1.,
        value_weight: float = 0.5,
        aux_weight: float = 0.5,
        entropy_weight: 'float | Tensor' = 1e-3,
        aux_task: AuxTask = None,
        log_dir: str = 'runs',
        bias_returns: bool = False,
        accelerate: bool = False
    ):
        if ckpt_tracker.model is None or ckpt_tracker.optimizer is None:
            raise AttributeError('Both model and optimizer must be pre-assigned.')

        if batch_size is None:
            batch_size = n_actors

        self.n_envs = n_envs
        self.n_actors = n_actors
        self.n_actors_per_env = n_actors // n_envs
        self.multi_agent = self.n_actors_per_env != 1
        self.n_envs_per_batch, leftover_samples = divmod(batch_size, self.n_actors_per_env)

        if leftover_samples:
            raise ValueError(f'Batch size ({batch_size}) incompatible with num. of actors ({self.n_actors_per_env}).')

        self.env_step = env_step
        self.ckpter = ckpt_tracker
        self.model: MultiActorCritic = ckpt_tracker.model
        self.optimizer: MultiOptimizer = ckpt_tracker.optimizer
        self.scheduler = scheduler

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, ckpt_tracker.model_name))
        self.write = self.writer.add_scalar

        self.n_epochs = n_epochs
        self.log_interval = log_epoch_interval
        self.checkpoint_interval = ckpt_epoch_interval
        self.branch_interval = branch_epoch_interval

        self.n_rollout_steps = n_rollout_steps
        self.n_truncated_steps = n_truncated_steps
        self.n_passes_per_step = n_passes_per_step
        self.buffer_size = buffer_size if buffer_size is not None else n_rollout_steps

        self.main_buffer = ExperienceBuffer(self.buffer_size)
        self.aux_task = aux_task

        if hasattr(discount_gammas, '__len__'):
            discount_gammas = torch.tensor((discount_gammas,), dtype=torch.float32, device=self.ckpter.device)

        self.discount_gammas = discount_gammas
        self.trace_lambda = trace_lambda
        self.bias_returns = bias_returns

        self.improv_bounds = 1. / (1. + clip_ratio), 1. + clip_ratio
        self.decline_bounds = 1. / (1. + 2*clip_ratio), 1. + 2*clip_ratio

        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.aux_weight = aux_weight
        self.entropy_weight = entropy_weight

        self.stats = {k: torch.tensor(0., device=self.ckpter.device) for k in self.STAT_KEYS}

        if aux_task:
            for k in aux_task.STAT_KEYS:
                self.stats[k] = torch.tensor(0., device=self.ckpter.device)

        self.reward = self.stats['Env/reward']
        self.score = 0.
        self.val = 0.
        self.lr = 0.

        self.accelerate = accelerate
        self.accel_graph = None
        self.update_accel = None

    def run(self):
        """
        Step the model and env. until enough experiences are recorded to update
        model params., repeating the loop for a given number of epochs
        and occasionally making a checkpoint and logging metrics.
        """

        # Initial obs. and mem.
        mem = self.model.init_mem(self.n_actors, self.n_envs)
        obs = self.env_step()[0]

        starting_step = epoch_step = self.ckpter.meta['epoch_step']
        last_log_step = starting_step
        starting_time = perf_counter()
        n_updates = 0

        for epoch_step in range(starting_step+1, self.n_epochs+1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time
            load_factor = 2. - self.main_buffer.load_ratio()

            remaining_time = min(
                int(running_time * load_factor * (self.n_epochs-epoch_step+1) / max(1, epoch_step-1-starting_step)),
                self.MAX_DISP_SECONDS)

            self.print_progress(progress, remaining_time, epoch_step)

            # Update exp. buffer
            self.main_buffer.clear(self.n_rollout_steps)

            if self.aux_task:
                self.aux_task.clear()

            if self.main_buffer:
                mem = self.main_buffer.batches[0]['mem']

            with torch.no_grad():
                for batch in self.main_buffer.batches:
                    mem = self.recollect(batch, mem)

                for _ in range(self.n_rollout_steps):
                    obs, mem = self.collect(obs, mem)

                if self.policy_weight:
                    self.label(obs, mem)

            # Update print-out info
            n_env_steps = (epoch_step - last_log_step) * self.n_rollout_steps
            self.score = self.stats.get('Env/score', self.reward).item() / n_env_steps

            # Shuffle sequences of batches
            if self.multi_agent:
                buffer = self.main_buffer.shuffle(
                    self.ckpter.rng,
                    seq_length=self.n_truncated_steps,
                    n_in_chunks=self.n_envs,
                    n_out_chunks=self.n_envs_per_batch)

            else:
                buffer = self.main_buffer.shuffle(self.ckpter.rng, seq_length=self.n_truncated_steps)

            # Iterate over sequences of batches and update the model with epochwise TBPTT
            for seq in buffer.iter_slices(self.n_truncated_steps):
                for _ in range(self.n_passes_per_step):

                    # CUDA graph acceleration
                    if self.accelerate:

                        # Flatten content of batches into a list of tensors to pass to the graph
                        full_input_list = [t for b in seq for t in b.to_list()]

                        # Capture computational graph
                        if self.accel_graph is None:
                            self.accel_update(seq, full_input_list)

                        self.update_accel(full_input_list)

                    # Main update
                    elif self.policy_weight:
                        self.update(seq)

                    # Auxiliary update
                    if self.aux_task and self.aux_task.offline:
                        self.aux_task.update(seq, self.stats)

                n_updates += self.n_passes_per_step

            self.scheduler.step()
            self.lr += self.scheduler.lr
            self.val += self.scheduler.value

            # Log running metrics and perf. score
            if self.log_interval and not epoch_step % self.log_interval:
                self.log(epoch_step, n_updates)

                last_log_step = epoch_step
                n_updates = 0

            # Save model params. and training state
            if self.branch_interval and not epoch_step % self.branch_interval:
                self.checkpoint(epoch_step, branch=True)

            elif self.checkpoint_interval and not epoch_step % self.checkpoint_interval:
                self.checkpoint(epoch_step)

        self.checkpoint(epoch_step)
        self.writer.close()

    def print_progress(self, progress: float, remaining_time: float, epoch_step: int):
        print(
            f'\rEpoch {epoch_step} of {self.n_epochs} ({progress:.2f}) | '
            f'ETA: {str(timedelta(seconds=remaining_time))} | '
            f'Score: {self.score: .3f}   ',
            end='')

    def log(self, epoch_step: int, n_updates: int):
        n_update_steps = n_updates * self.n_truncated_steps
        n_env_steps = self.log_interval * self.n_rollout_steps
        env_step = epoch_step * self.n_rollout_steps

        if self.aux_task and self.aux_task.n_update_steps:
            n_aux_update_steps = self.aux_task.n_update_steps
            self.aux_task.n_update_steps = 0

        else:
            n_aux_update_steps = n_update_steps

        for key, val in self.stats.items():
            if key.startswith('GAE'):
                den = self.log_interval

            elif key.startswith('Env'):
                den = n_env_steps

            elif key.startswith('Aux'):
                den = n_aux_update_steps

            else:
                den = n_update_steps

            self.write(key, val.item() / den, env_step)
            val.zero_()

        self.write('Opt/lr', self.lr / self.log_interval, env_step)
        self.write('Opt/val', self.val / self.log_interval, env_step)
        self.lr = 0.
        self.val = 0.

    def checkpoint(self, epoch_step: int, branch: bool = False):
        update_step = self.scheduler.step_ctr
        ckpt_increment = 1 if branch else 0

        self.ckpter.checkpoint(epoch_step, update_step, ckpt_increment, self.score)

    def collect(self, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]') -> 'tuple[tuple[Tensor, ...], ...]':

        # Step actors
        data, aux, mem = self.model.collect(obs, mem, None)

        # Step envs.
        obs, env_data, log_info = self.env_step(*self.model.unwrap_sample(data['act'], aux))

        data.update(env_data)
        nrst = data['nrst']
        rwd = data['rwd']

        # Reset memory if any terminal states are reached
        if not nrst.all():
            mem = self.model.reset_mem(mem, nrst)

        # Add batch to buffers
        d = TensorDict(data)

        self.main_buffer.append(d)

        if self.aux_task:
            self.aux_task.collect(d, obs, mem)

        # Add env. info. to logged metrics
        for key, val in log_info.items():
            key = f'Env/{key}'

            if key not in self.stats:
                self.stats[key] = torch.tensor(0., device=self.ckpter.device)

            self.stats[key] += val

        self.stats['Env/reward'] += rwd.sum(-1).mean()
        self.stats['Env/resets'] += self.n_actors - nrst.sum()

        return obs, mem

    def recollect(self, b: TensorDict, mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':

        # Step actors
        data, _, mem = self.model.collect(b['obs'], mem, b['act'])

        # Update value estimates
        b['val'] = data['val']

        # Update mem. in-place, so tensors/views in aux. buffers can benefit too
        for t, new_t in zip(b['mem'], data['mem']):
            t.copy_(new_t)

        # Reset memory if any terminal states are reached
        if not b['nrst'].all():
            mem = self.model.reset_mem(mem, b['nrst'])

        return mem

    def label(self, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):

        # Perform an additional critic pass to get the final values used in GAE
        values = self.model.collect(obs, mem, None)[0]['val']

        # Set or update return targets and advantages
        adv_mean, adv_std = self.main_buffer.label(
            values, self.discount_gammas, self.trace_lambda, self.n_actors_per_env, self.bias_returns)

        self.stats['GAE/adv_mean'] += adv_mean
        self.stats['GAE/adv_std'] += adv_std

    def update(self, batches: 'list[TensorDict]'):
        stats = self.stats
        running_loss = 0.
        mem = batches[0]['mem']

        self.optimizer.zero_grad()

        for b in batches:
            data = self.model(b['obs'], mem, b['act'])
            mem = self.model.reset_mem(data['mem'], b['nrst'])

            # Policy
            act: Distribution = data['act']
            old_act: Distribution = self.model.get_distr(b['args'])

            ratio = (act.log_prob(*b['act']) - old_act.log_prob(*b['act'])).exp()
            advp = b['advp'] if ratio.ndim == 1 else b['advp'].unsqueeze(-1)

            policy_loss = torch.minimum(
                advp * ratio.clamp(*self.improv_bounds),
                advp * ratio.clamp(*self.decline_bounds)).mean()

            # Value
            if self.multi_agent:
                advj: Distribution = data['advj']
                valj: Distribution = data['valj']
                vali: Distribution = data['vali']
                vals = valj, vali

                target_adv_loss = advj.log_prob(b['advt']).mean()
                joint_value_loss = valj.log_prob(b['retj']).mean()
                indiv_value_loss = vali.log_prob(b['reti']).mean()
                value_loss = target_adv_loss + joint_value_loss + indiv_value_loss

            else:
                val: Distribution = data['val']
                vals = val,

                target_adv_loss = 0.
                joint_value_loss = 0.
                indiv_value_loss = val.log_prob(b['ret']).mean()
                value_loss = indiv_value_loss

            # Auxiliary
            if self.aux_task and self.aux_task.online:
                aux_loss = self.aux_task.loss(b, act, vals, data['aux'], stats)

            else:
                aux_loss = 0.

            # Entropy
            entropy = act.entropy.mean()

            # Total
            full_loss = (
                self.aux_weight * aux_loss
                - self.policy_weight * policy_loss
                - self.value_weight * value_loss
                - self.entropy_weight * entropy)

            running_loss = running_loss + full_loss

            # Stats for logging
            with torch.no_grad():
                stats['Out/act_mean'] += act.mean.mean()
                stats['Out/act_std'] += act.dev.mean()
                stats['Out/val_mean'] += b['val'].sum(-1).mean()
                stats['Main/loss'] += full_loss
                stats['Main/policy'] -= policy_loss
                stats['Main/value'] -= value_loss
                stats['Val/adv'] -= target_adv_loss
                stats['Val/tot'] -= joint_value_loss
                stats['Val/ind'] -= indiv_value_loss
                stats['Aux/loss'] += aux_loss
                stats['Main/entropy'] += entropy
                stats['Main/ratio_diff'] += (ratio - 1.).abs().mean()

        # Average loss over N time steps for TBPTT
        # NOTE: https://r2rt.com/styles-of-truncated-backpropagation.html
        loss = running_loss / self.n_truncated_steps
        loss.backward()

        self.optimizer.step()

    def accel_update(self, batches: 'list[TensorDict]', inputs: 'list[Tensor]'):

        # Warmup on side stream
        s = cuda.Stream()
        s.wait_stream(cuda.current_stream())

        with cuda.stream(s):

            # Get current state
            stats_values = [v.item() for v in self.stats.values()]

            model_state_bytes = BytesIO()
            optim_state_bytes = BytesIO()

            torch.save(self.model.state_dict(), model_state_bytes)
            torch.save(self.optimizer.state_dict(), optim_state_bytes)

            model_state_bytes.seek(0)
            optim_state_bytes.seek(0)

            # Warmup steps
            for _ in range(3):
                self.update(batches)

            # Restore state before warmup
            for v, v_ in zip(self.stats.values(), stats_values):
                v.fill_(v_)

            self.model.load_state_dict(torch.load(model_state_bytes))
            self.optimizer.load_state_dict(torch.load(optim_state_bytes))

            model_state_bytes.close()
            optim_state_bytes.close()

        cuda.current_stream().wait_stream(s)

        # Restore input structure and relay as actual args.
        batch_ref = batches[0]

        def update_accel(inputs: 'list[Tensor]'):
            input_len = len(inputs) // self.n_truncated_steps
            batches = [batch_ref.from_list(inputs[i:i+input_len]) for i in range(0, len(inputs), input_len)]

            self.update(batches)

        # Capture computational graph
        self.update_accel, self.accel_graph = capture_graph(
            update_accel,
            inputs,
            warmup_tensor_list=(),
            single_input=True)

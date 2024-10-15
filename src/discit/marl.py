"""Multi-agent reinforcement learning"""

import os
from abc import ABC, abstractmethod
from datetime import timedelta
from time import perf_counter
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from .data import ExperienceBuffer, TensorDict
from .distr import Distribution
from .optim import MultiOptimizer, MultiScheduler
from .track import CheckpointTracker


class MultiActorCritic(Module, ABC):
    def __init__(self):
        Module.__init__(self)

    def init_mem(self, n_envs: int, n_actors: int) -> 'tuple[Tensor, ...]':
        return ()

    def reset_mem(self, mem: 'tuple[Tensor, ...]', nonreset_mask: Tensor) -> 'tuple[Tensor, ...]':
        return ()

    @abstractmethod
    def get_distr(self, args: 'tuple[Tensor, ...]') -> Distribution:
        ...

    def unwrap_sample(self, sample: 'tuple[Tensor, ...]', aux: 'tuple[Tensor, ...]') -> 'tuple[Tensor | None, ...]':
        return sample[0], None

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
        act: 'tuple[Tensor, ...]'
    ) -> 'dict[str, Distribution | tuple[Distribution, ...] | Tensor | tuple[Tensor, ...]]':
        ...


class AuxTask(ABC):
    STAT_KEYS = ()

    def __init__(self, online: bool = False, offline: bool = False):
        self.online = online
        self.offline = offline

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def collect(self, batch: TensorDict):
        ...

    @abstractmethod
    def update(self, batches: 'list[TensorDict]', stats: 'dict[str, Tensor]'):
        ...

    @abstractmethod
    def loss(
        self,
        batch: TensorDict,
        act: Distribution,
        val: 'tuple[Distribution, ...]',
        aux: 'tuple[Distribution, ...]',
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
        'GAE/adv_std',
        'GAE/imp_mean')

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
        buffer_size: int = None,
        batch_size: int = None,
        aux_task: AuxTask = None,
        discount_gammas: 'float | tuple[float, ...]' = 0.99,
        trace_lambdas: 'float | tuple[float, float]' = 0.99,
        clip_ratio: float = 0.2,
        policy_weight: float = 1.,
        value_weight: float = 0.5,
        aux_weight: float = 0.,
        entropy_weight: 'float | Tensor' = 1e-3,
        log_dir: str = 'runs'
    ):
        if ckpt_tracker.model is None or ckpt_tracker.optimizer is None:
            raise AttributeError('Both model and optimizer must be pre-assigned.')

        if batch_size is None:
            batch_size = n_actors

        self.n_envs = n_envs
        self.n_actors = n_actors
        self.n_actors_per_env = n_actors // n_envs
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
        self.buffer_size = buffer_size if buffer_size is not None else n_rollout_steps

        self.main_buffer = ExperienceBuffer(self.buffer_size)
        self.aux_task = aux_task

        if hasattr(discount_gammas, '__len__'):
            discount_gammas = torch.tensor((discount_gammas,), dtype=torch.float32, device=self.ckpter.device)

        self.discount_gammas = discount_gammas
        self.trace_lambdas = trace_lambdas

        self.clip_min = torch.tensor(1. / (1. + clip_ratio), device=self.ckpter.device)
        self.log_clip_max = torch.tensor(1. + clip_ratio, device=self.ckpter.device).log()

        self.policy_weight = -policy_weight
        self.value_weight = -value_weight
        self.aux_weight = -aux_weight
        self.entropy_weight = entropy_weight

        self.init_imp_weight = torch.zeros((n_actors, 1), device=self.ckpter.device)
        self.init_imp_max = (1. + clip_ratio) * torch.ones(n_actors, device=self.ckpter.device)

        self.stats = {k: torch.tensor(0., device=self.ckpter.device) for k in self.STAT_KEYS}

        if aux_task:
            for k in aux_task.STAT_KEYS:
                self.stats[k] = torch.tensor(0., device=self.ckpter.device)

        self.reward = self.stats['Env/reward']
        self.score = 0.
        self.val = 0.
        self.lr = 0.

    def run(self):
        """
        Step the model and env. until enough experiences are recorded to update
        model params., repeating the loop for a given number of epochs
        and occasionally making a checkpoint and logging metrics.
        """

        # Initial obs. and mem.
        mem = self.model.init_mem(self.n_envs, self.n_actors)
        obs = self.env_step()[0]

        starting_step = epoch_step = self.ckpter.meta['epoch_step']
        last_log_step = starting_step
        starting_time = perf_counter()
        n_updates = 0

        for epoch_step in range(starting_step+1, self.n_epochs+1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time

            remaining_time = min(
                int(running_time * (self.n_epochs - epoch_step + 1) / max(1, epoch_step - 1 - starting_step)),
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

            n_env_steps = (epoch_step - last_log_step) * self.n_rollout_steps
            self.score = self.stats.get('Env/score', self.reward).item() / n_env_steps

            # Update model
            buffer = self.main_buffer.shuffle(
                self.ckpter.rng,
                seq_length=self.n_truncated_steps,
                n_in_chunks=self.n_envs,
                n_out_chunks=self.n_envs_per_batch)

            for seq in buffer.iter_slices(self.n_truncated_steps):
                if self.policy_weight:
                    self.update(seq)

                if self.aux_task and self.aux_task.offline:
                    self.aux_task.update(seq, self.stats)

                n_updates += 1

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

        for key, val in self.stats.items():
            if key.startswith('GAE'):
                den = self.log_interval

            elif key.startswith('Env'):
                den = n_env_steps

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
        data, aux_belief, mem = self.model.collect(obs, mem, None)

        # Step envs.
        obs, env_data, info = self.env_step(*self.model.unwrap_sample(data['act'], aux_belief))

        data.update(env_data)
        data['imp'] = self.init_imp_weight
        data['max'] = self.init_imp_max

        nrst = data['nrst']
        rwd = data['rwd']

        # Add batch to buffers
        d = TensorDict(data)

        self.main_buffer.append(d)

        if self.aux_task:
            self.aux_task.collect(d)

        # Reset memory if any terminal states are reached
        if not nrst.all():
            mem = self.model.reset_mem(mem, nrst)

        # Add env. info. to logged metrics
        for key, val in info.items():
            key = f'Env/{key}'

            if key not in self.stats:
                self.stats[key] = torch.tensor(0., device=self.ckpter.device)

            self.stats[key] += val

        self.stats['Env/reward'] += rwd.sum(-1).mean()
        self.stats['Env/resets'] += self.n_actors - nrst.sum()

        return obs, mem

    def recollect(self, b: TensorDict, mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        # Step actor
        data, _, mem = self.model.collect(b['obs'], mem, sample=b['act'])

        # Update batch
        act = self.model.get_distr(data['args'])
        old_act = self.model.get_distr(b['args'])

        imp = act.log_prob(*b['act']) - old_act.log_prob(*b['act'])
        clip_max = (imp + self.log_clip_max).exp()

        b['max'] = clip_max
        b['imp'] = imp
        b['val'] = data['val']

        # Update act. and mem. in-place, so tensors/views in aux. buffers can benefit too
        for t, new_t in zip(b['args'], data['args']):
            t.copy_(new_t)

        for t, new_t in zip(b['mem'], data['mem']):
            t.copy_(new_t)

        # Reset memory if any terminal states are reached
        if not b['nrst'].all():
            mem = self.model.reset_mem(mem, b['nrst'])

        return mem

    def label(self, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):

        # Perform an additional critic pass to get the final values used in GAE
        values = self.model.collect(obs, mem, None)[0]['val']

        adv_mean, adv_std, imp_mean = self.main_buffer.multilabel(
            values, self.discount_gammas, self.trace_lambdas, self.n_actors_per_env)

        self.stats['GAE/adv_mean'] += adv_mean
        self.stats['GAE/adv_std'] += adv_std
        self.stats['GAE/imp_mean'] += imp_mean

    def update(self, batches: 'list[TensorDict]'):
        stats = self.stats
        running_loss = 0.
        mem = batches[0]['mem']

        self.optimizer.zero_grad()

        for b in batches:
            data = self.model(b['obs'], mem, b['act'])

            act: Distribution = data['act']
            adv: Distribution = data['advj']
            val_tot: Distribution = data['valj']
            val_ind: Distribution = data['vali']
            aux: 'tuple[Distribution, ...]' = data['aux']

            mem = self.model.reset_mem(data['mem'], b['nrst'])

            # Policy
            old_act: Distribution = self.model.get_distr(b['args'])
            ratio = (act.log_prob(*b['act']) - old_act.log_prob(*b['act'])).exp()

            policy_loss = torch.minimum(
                b['adv_pi'] * ratio,
                b['adv_pi'] * ratio.clamp(self.clip_min, b['max'])).mean()

            # Value
            target_adv_loss = adv.log_prob(b['adv_tar']).mean()
            joint_value_loss = val_tot.log_prob(b['ret_joint']).mean()
            indiv_value_loss = val_ind.log_prob(b['ret_indiv']).mean()
            value_loss = target_adv_loss + joint_value_loss + indiv_value_loss

            # Auxiliary
            if self.aux_task and self.aux_task.online:
                aux_loss = self.aux_task.loss(b, act, (val_tot, val_ind), aux, stats)

            else:
                aux_loss = 0.

            # Entropy
            entropy = act.entropy.mean()

            # Total
            full_loss = (
                self.policy_weight * policy_loss
                + self.value_weight * value_loss
                + self.aux_weight * aux_loss
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
                stats['Aux/loss'] -= aux_loss
                stats['Main/entropy'] += entropy
                stats['Main/ratio_diff'] += (ratio - 1.).abs().mean()

        # Average loss over N time steps for TBPTT
        # NOTE: https://r2rt.com/styles-of-truncated-backpropagation.html
        loss = running_loss / self.n_truncated_steps
        loss.backward()

        self.optimizer.step()

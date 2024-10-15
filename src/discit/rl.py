"""Reinforcement learning"""

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
from .optim import CoeffScheduler, LRScheduler
from .track import CheckpointTracker


class ActorCritic(Module, ABC):
    def __init__(self):
        Module.__init__(self)

    @abstractmethod
    def init_mem(self, n_actors: int) -> 'tuple[Tensor, ...]':
        ...

    @abstractmethod
    def reset_mem(self, mem: 'tuple[Tensor, ...]', nonreset_mask: Tensor) -> 'tuple[Tensor, ...]':
        ...

    @abstractmethod
    def get_distr(self, args: 'Tensor | tuple[Tensor, ...]', from_raw: bool) -> Distribution:
        ...

    def unwrap_sample(self, sample: 'tuple[Tensor, ...]', aux: 'tuple[Tensor, ...]') -> 'tuple[Tensor | None, ...]':
        return sample[0], None

    @abstractmethod
    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        encode: bool
    ) -> 'tuple[Tensor, Tensor, Tensor | None, tuple[Tensor, ...], tuple[Tensor, ...]]':
        ...

    @abstractmethod
    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        detach: bool
    ) -> 'tuple[Distribution, Distribution, tuple[Distribution], tuple[Tensor, ...]]':
        ...


class PPG:
    """Implementation of phasic policy gradient algorithm focusing on recurrent models."""

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
        'Main/vaux',
        'Main/entropy',
        'Main/ratio_diff',
        'GAE/adv_mean',
        'GAE/adv_std',
        'Aux/loss',
        'Aux/value',
        'Aux/vaux',
        'Aux/kl_div',
        'Aux/imp')

    def __init__(
        self,
        env_step: Callable[
            ['Tensor | None', 'Tensor | None'],
            'tuple[tuple[Tensor, ...], Tensor, Tensor, dict[str, float]]'],
        ckpt_tracker: CheckpointTracker,
        scheduler: LRScheduler,
        n_actors: int,
        n_epochs: int,
        log_epoch_interval: int = 1,
        ckpt_epoch_interval: int = 3,
        branch_epoch_interval: int = 10,
        n_rollout_steps: int = 256,
        n_truncated_steps: int = 16,
        n_passes_per_batch: int = 10,
        batch_size: int = None,
        n_main_iters: int = 8,
        n_aux_iters: int = 6,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        policy_weight: float = 1.,
        value_weight: float = 0.5,
        aux_weight: float = 0.,
        entropy_weight: 'float | CoeffScheduler' = 1e-3,
        log_dir: str = 'runs',
        accelerate: bool = True,
        replay_rollout: bool = True,
        update_returns: bool = True,
        detach_critic: bool = True
    ):
        if ckpt_tracker.model is None or ckpt_tracker.optimizer is None:
            raise AttributeError('Both model and optimizer must be pre-assigned.')

        if batch_size is None:
            batch_size = n_actors

        self.reshape_buffer = batch_size != n_actors

        if self.reshape_buffer:
            if n_truncated_steps > 1:
                raise NotImplementedError('Cannot reshape buffer for sequential actors.')

            if replay_rollout:
                raise NotImplementedError('Cannot replay rollout for reshaped buffer.')

            if update_returns:
                raise NotImplementedError('Cannot update returns for reshaped buffer.')

            self.n_batches_to_reshape, leftover_samples = \
                divmod(batch_size, n_actors) if batch_size > n_actors else divmod(-n_actors, batch_size)

            if leftover_samples:
                raise ValueError(f'Batch size ({batch_size}) incompatible with num. of actors ({n_actors}).')

        else:
            self.n_batches_to_reshape = None

        if n_rollout_steps % n_truncated_steps:
            raise ValueError(
                f'Num. of rollout steps ({n_rollout_steps}) incompatible with '
                f'num. of truncated steps ({n_truncated_steps}).')

        self.env_step = env_step
        self.ckpter = ckpt_tracker
        self.model: ActorCritic = ckpt_tracker.model
        self.optimizer = ckpt_tracker.optimizer
        self.scheduler = scheduler

        self.accelerate = accelerate
        self.accel_graphs = {}
        self.update_main_single_accel = None
        self.update_aux_single_accel = None

        self.replay_rollout = replay_rollout
        self.update_returns = update_returns
        self.detach_critic = detach_critic

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, ckpt_tracker.model_name))
        self.write = self.writer.add_scalar

        self.n_epochs = n_epochs
        self.log_interval = log_epoch_interval
        self.checkpoint_interval = ckpt_epoch_interval
        self.branch_interval = branch_epoch_interval

        self.n_rollout_steps = n_rollout_steps
        self.n_truncated_steps = n_truncated_steps
        self.n_passes_per_batch = n_passes_per_batch
        self.n_actors = n_actors
        self.n_main_iters = n_main_iters
        self.n_aux_iters = n_aux_iters

        if hasattr(discount_factor, '__len__'):
            discount_factor = torch.tensor((discount_factor,), dtype=torch.float32, device=self.ckpter.device)

        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_bounds = 1. / (1. + clip_ratio), 1. + clip_ratio
        self.policy_weight = -policy_weight
        self.value_weight = -value_weight
        self.aux_weight = -aux_weight

        if isinstance(entropy_weight, CoeffScheduler):
            self.entropy_scheduler = entropy_weight
            self.entropy_weight = self.entropy_scheduler.value

        else:
            self.entropy_scheduler = None
            self.entropy_weight = entropy_weight

        self.in_main_phase = False
        self.main_buffer = ExperienceBuffer(n_rollout_steps)
        self.aux_buffer = ExperienceBuffer(n_rollout_steps * n_main_iters)

        self.score = 0.
        self.lr = 0.

        self.ratio_diff = torch.tensor(0., device=self.ckpter.device)
        self.init_imp_weight = torch.ones((n_actors, 1), device=self.ckpter.device)

        self.stats = {k: torch.tensor(0., device=self.ckpter.device) for k in self.STAT_KEYS}
        self.reward = self.stats['Env/reward']

    def run(self):
        """
        Step the model and env. until enough experiences are recorded to update
        model params., repeating the loop for a given number of epochs
        and occasionally making a checkpoint and logging metrics.
        """

        # Initial obs. and mem.
        mem = self.model.init_mem(self.n_actors)
        obs = self.env_step()[0]

        starting_step = epoch_step = self.ckpter.meta['epoch_step']
        last_log_step = starting_step
        starting_time = perf_counter()

        for epoch_step in range(starting_step+1, self.n_epochs+1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time

            remaining_time = min(
                int(running_time * (self.n_epochs - epoch_step + 1) / max(1, epoch_step - 1 - starting_step)),
                self.MAX_DISP_SECONDS)

            # Main phase
            self.in_main_phase = True

            for iter_step in range(1, self.n_main_iters+1):
                self.print_progress(progress, remaining_time, epoch_step, iter_step)

                obs, mem = self.collect(obs, mem)

                if self.reshape_buffer:
                    self.main_buffer = self.main_buffer.restack(self.n_batches_to_reshape, self.ckpter.rng)

                for update_step in range(self.n_passes_per_batch):
                    if self.update_returns and update_step != 0:
                        self.recollect(obs)

                    updated_mem = self.update_main(iter_step + self.n_main_iters * (epoch_step - last_log_step - 1))

                if not self.reshape_buffer:
                    mem = updated_mem

                self.main_buffer.clear()

            # Aux phase
            self.in_main_phase = False

            if self.reshape_buffer and self.n_aux_iters > 0:
                self.aux_buffer = self.aux_buffer.restack(self.n_batches_to_reshape, self.ckpter.rng)

            for iter_step in range(1, self.n_aux_iters+1):
                self.print_progress(progress, remaining_time, epoch_step, iter_step)

                if self.update_returns:
                    self.recollect(obs, update_act=iter_step == 1)

                updated_mem = self.update_aux()

            if not self.reshape_buffer:
                mem = updated_mem

            self.aux_buffer.clear()

            # Log running metrics and perf. score
            if self.log_interval and not epoch_step % self.log_interval:
                self.log(epoch_step)
                last_log_step = epoch_step

            # Save model params. and training state
            if self.branch_interval and not epoch_step % self.branch_interval:
                self.checkpoint(epoch_step, branch=True)

            elif self.checkpoint_interval and not epoch_step % self.checkpoint_interval:
                self.checkpoint(epoch_step)

        self.checkpoint(epoch_step)
        self.writer.close()

    def print_progress(self, progress: float, remaining_time: float, epoch_step: int, iter_step: int):
        print(
            f'\rEpoch {epoch_step} of {self.n_epochs} ({progress:.2f}) | '
            f'Iter. {iter_step} of {self.n_main_iters if self.in_main_phase else self.n_aux_iters} '
            f'({"main" if self.in_main_phase else "aux"}) | '
            f'ETA: {str(timedelta(seconds=remaining_time))} | '
            f'Score: {self.score:.4f}        ',
            end='')

    def log(self, epoch_step: int):
        n_env = self.n_rollout_steps * self.n_main_iters * self.log_interval
        n_main = n_env * self.n_passes_per_batch
        n_aux = n_env * self.n_aux_iters
        n_gae = self.n_main_iters * self.log_interval
        n_upd = (n_main + n_aux) // self.n_truncated_steps

        env_step = epoch_step * n_env // self.log_interval

        for key, val in self.stats.items():
            if key.startswith('Aux'):
                if self.n_aux_iters == 0:
                    continue

                den = n_aux

            elif key.startswith('GAE'):
                den = n_gae

            elif key.startswith('Env'):
                den = n_env

            else:
                den = n_main

            self.write(key, val.item() / den, env_step)
            val.zero_()

        self.write('Opt/lr', self.lr / n_upd, env_step)
        self.lr = 0.

    def checkpoint(self, epoch_step: int, branch: bool = False):
        update_step = self.scheduler.step_ctr
        ckpt_increment = 1 if branch else 0

        self.ckpter.checkpoint(epoch_step, update_step, ckpt_increment, self.score)

    def collect(self, obs: Tensor, mem: Tensor) -> 'tuple[tuple[Tensor, ...], ...]':
        with torch.no_grad():
            for _ in range(self.n_rollout_steps):

                # Step actor
                act_out, val_mean, aux_belief, obs_enc, new_mem = self.model.collect(obs, mem, encode=True)

                act = self.model.get_distr(act_out, from_raw=True)
                act_sample = act.sample()

                # Step env.
                obs, rew, rst, *val_aux, info = self.env_step(*self.model.unwrap_sample(act_sample, aux_belief))
                nonrst = 1. - rst

                # Add batch to buffers
                d = TensorDict({
                    'act': act_sample,
                    'args': act.args,
                    'val': val_mean,
                    'obs': obs_enc,
                    'mem': mem,
                    'rwd': rew,
                    'imp': self.init_imp_weight,
                    'nrst': nonrst})

                if val_aux:
                    d['vaux'] = tuple(val_aux)

                self.main_buffer.append(d)
                self.aux_buffer.append(d)

                # Reset memory if any terminal states are reached
                mem = new_mem

                if not nonrst.all():
                    mem = self.model.reset_mem(mem, nonrst)

                # Add env. info. to logged metrics
                for key, val in info.items():
                    key = f'Env/{key}'

                    if key not in self.stats:
                        self.stats[key] = torch.tensor(0., device=self.ckpter.device)

                    self.stats[key] += val

                self.stats['Env/reward'] += rew.sum(-1).mean()
                self.stats['Env/resets'] += rst.sum()

            # Perform an additional critic pass to get the final values used in GAE
            _, values, _, _ = self.model.collect(obs, mem, encode=True)

            adv_mean, adv_std = self.main_buffer.label(values, self.discount_factor, self.gae_lambda)

            self.stats['GAE/adv_mean'] += adv_mean
            self.stats['GAE/adv_std'] += adv_std

        return obs, mem

    def recollect(self, final_obs: 'tuple[Tensor, ...]', update_act: bool = False):
        buffer = self.main_buffer if self.in_main_phase else self.aux_buffer
        mem = self.aux_buffer.batches[0]['mem']

        with torch.no_grad():
            for b in buffer.batches:

                # Step actor
                act_out, val_mean, _, _, new_mem = self.model.collect(b['obs'], mem, encode=False)

                # Update batch
                if self.in_main_phase or update_act:
                    act = self.model.get_distr(act_out, from_raw=True)
                    old_act = self.model.get_distr(b['args'], from_raw=False)
                    clipped_imp = (act.log_prob(*b['act']) - old_act.log_prob(*b['act'])).clamp(None, 0.).exp()

                    b['imp'] = clipped_imp.unsqueeze(-1)

                    if update_act:
                        b['act'] = act.args

                b['val'] = val_mean
                b['mem'] = mem

                # Reset memory if any terminal states are reached
                mem = new_mem

                if not b['nrst'].all():
                    mem = self.model.reset_mem(mem, b['nrst'])

            # Perform an additional critic pass to get the final values used in GAE
            _, values, _, _ = self.model.collect(final_obs, mem, encode=True)

            # Update target returns
            buffer.label(values, self.discount_factor, self.gae_lambda, skip_std=not self.in_main_phase)

    def update_main(self, iter_num: int) -> 'tuple[Tensor, ...]':
        """
        Iterate over sequences of batches in a rollout and update
        model params. according to the main objective with epochwise TBPTT.
        """

        mem = None

        for batches in self.main_buffer.iter_slices(self.n_truncated_steps):
            self.ratio_diff.zero_()

            # Standard truncation
            if mem is None or not self.replay_rollout:
                mem = batches[0]['mem']

            # Burn-in on full trajectory up to this point in the rollout
            # NOTE: State correlations must be managed externally
            else:
                mem = [m.detach() for m in mem]

            # CUDA graph acceleration
            # TODO: Mem. out. at first iter. of first epoch seems different from collector with same inputs and params.
            if self.accelerate:

                # Flatten content of batches into a list of tensors to pass to the graph
                full_input_list = [t for b in batches for t in b.to_list()]
                full_input_list.extend(mem)

                # Capture computational graph
                if self.update_main_single_accel is None:
                    self.accel_main(batches, mem, full_input_list)

                mem = self.update_main_single_accel(full_input_list)

            else:
                self.optimizer.zero_grad()
                mem = self.update_main_single(batches, mem)

            self.scheduler.step(self.ratio_diff.item() / self.n_truncated_steps)
            self.lr += self.scheduler.lr

            if self.entropy_scheduler is not None:
                self.entropy_scheduler.step()

        # Update print-out info
        self.score = self.stats.get('Env/score', self.reward).item() / (self.n_rollout_steps * iter_num)

        return tuple([m.detach() for m in mem])

    def update_main_single(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        stats = self.stats
        running_loss = 0.

        for batch in batches:
            act, val, *aux, mem = self.model(batch['obs'], mem, detach=self.detach_critic)

            act: Distribution
            val: Distribution
            aux: 'list[Distribution]'

            mem = self.model.reset_mem(mem, batch['nrst'])

            # Policy
            old_act: Distribution = self.model.get_distr(batch['args'], from_raw=False)

            act_log_prob = act.log_prob(*batch['act'])
            old_act_log_prob = old_act.log_prob(*batch['act'])

            ratio = (act_log_prob - old_act_log_prob).exp()

            policy_loss = torch.minimum(
                batch['adv'] * ratio,
                batch['adv'] * ratio.clamp(*self.clip_bounds)).mean()

            # Value
            # NOTE: No value loss clipping
            value_loss = val.log_prob(batch['ret']).mean()

            # Auxiliary
            aux_loss = 0.

            if 'vaux' in batch:
                for aux_i, ref_aux_i in zip(aux, batch['vaux']):
                    aux_loss = aux_loss + aux_i.log_prob(ref_aux_i).mean()

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
                ratio_diff = (ratio - 1.).abs().mean()
                self.ratio_diff += ratio_diff

                stats['Out/act_mean'] += act.mean.mean()
                stats['Out/act_std'] += act.dev.mean()
                stats['Out/val_mean'] += batch['val'].sum(-1).mean()
                stats['Main/loss'] += full_loss
                stats['Main/policy'] -= policy_loss
                stats['Main/value'] -= value_loss
                stats['Main/vaux'] -= aux_loss
                stats['Main/entropy'] += entropy
                stats['Main/ratio_diff'] += ratio_diff

        # Average loss over N time steps for TBPTT
        # NOTE: https://r2rt.com/styles-of-truncated-backpropagation.html
        loss = running_loss / self.n_truncated_steps
        loss.backward()

        self.optimizer.step()

        return mem

    def accel_main(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]', inputs: 'list[Tensor]'):

        # Warmup on side stream
        s = cuda.Stream()
        s.wait_stream(cuda.current_stream())

        with cuda.stream(s):

            # Get current state
            reward = self.reward.item()
            stats_values = [v.item() for v in self.stats.values()]

            model_state_bytes = BytesIO()
            optim_state_bytes = BytesIO()

            torch.save(self.model.state_dict(), model_state_bytes)
            torch.save(self.optimizer.state_dict(), optim_state_bytes)

            model_state_bytes.seek(0)
            optim_state_bytes.seek(0)

            # Warmup steps
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                self.update_main_single(batches, mem)

            # Restore state before warmup
            self.reward.fill_(reward)

            for v, v_ in zip(self.stats.values(), stats_values):
                v.fill_(v_)

            self.model.load_state_dict(torch.load(model_state_bytes))
            self.optimizer.load_state_dict(torch.load(optim_state_bytes))

            model_state_bytes.close()
            optim_state_bytes.close()

        cuda.current_stream().wait_stream(s)

        # Restore input structure and relay as actual args.
        n_mem_items = len(mem)
        batch_ref = batches[0]

        def update_main_single_accel(inputs: 'list[Tensor]') -> 'tuple[Tensor, ...]':
            if n_mem_items:
                inputs, mem = inputs[:-n_mem_items], inputs[-n_mem_items:]

            else:
                mem = []

            input_len = len(inputs) // self.n_truncated_steps
            batches = [batch_ref.from_list(inputs[i:i+input_len]) for i in range(0, len(inputs), input_len)]

            return self.update_main_single(batches, mem)

        # Capture computational graph
        self.optimizer.zero_grad(set_to_none=True)

        self.update_main_single_accel, self.accel_graphs['main'] = capture_graph(
            update_main_single_accel,
            inputs,
            warmup_tensor_list=(),
            single_input=True)

    def update_aux(self) -> 'tuple[Tensor, ...]':
        """
        Iterate over sequences of batches in combined rollouts and update
        model params. according to the auxiliary objective with epochwise TBPTT.
        """

        mem = None

        for batches in self.aux_buffer.iter_slices(self.n_truncated_steps):

            # Standard truncation
            if mem is None or not self.replay_rollout:
                mem = batches[0]['mem']

            # Burn-in on full trajectory up to this point in the rollout
            # NOTE: State correlations must be managed externally
            else:
                mem = tuple([m.detach() for m in mem])

            # CUDA graph acceleration
            if self.accelerate:

                # Flatten content of batches into a list of tensors to pass to the graph
                full_input_list = [t for b in batches for t in b.to_list()]
                full_input_list.extend(mem)

                # Capture computational graph
                if self.update_aux_single_accel is None:
                    self.accel_aux(batches, mem, full_input_list)

                mem = tuple(self.update_aux_single_accel(full_input_list))

            else:
                self.optimizer.zero_grad()
                mem = self.update_aux_single(batches, mem)

            self.scheduler.step()
            self.lr += self.scheduler.lr

            # TODO: Updating entropy coeff. in aux. phase has no immediate effect
            # Done only to avoid mismatch between LR and entropy scheduler args.
            if self.entropy_scheduler is not None:
                self.entropy_scheduler.step()

        return tuple([m.detach() for m in mem])

    def update_aux_single(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        stats = self.stats
        running_loss = 0.

        for batch in batches:
            act, val, *aux, mem = self.model(batch['obs'], mem, detach=False)

            act: Distribution
            val: Distribution
            aux: 'list[Distribution]'

            mem = self.model.reset_mem(mem, batch['nrst'])

            # KL divergence
            old_act: Distribution = self.model.get_distr(batch['args'], from_raw=False)

            kl_div = old_act.kl_div(act).mean()

            # Value
            # NOTE: No value loss clipping
            value_loss = val.log_prob(batch['ret']).mean()

            # Auxiliary
            aux_loss = 0.

            if 'vaux' in batch:
                for aux_i, ref_aux_i in zip(aux, batch['vaux']):
                    aux_loss = aux_loss + aux_i.log_prob(ref_aux_i).mean()

            # Total
            full_loss = self.value_weight * value_loss + self.aux_weight * aux_loss + kl_div
            running_loss = running_loss + full_loss

            # Stats for logging
            with torch.no_grad():
                stats['Aux/loss'] += full_loss
                stats['Aux/value'] -= value_loss
                stats['Aux/vaux'] -= aux_loss
                stats['Aux/kl_div'] += kl_div
                stats['Aux/imp'] += batch['imp'].mean()

        # Average loss over N time steps for TBPTT
        # NOTE: https://r2rt.com/styles-of-truncated-backpropagation.html
        loss = running_loss / self.n_truncated_steps
        loss.backward()

        self.optimizer.step()

        return mem

    def accel_aux(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]', inputs: 'list[Tensor]'):

        # NOTE: Repeating warmup here resulted in illegal CUDA memory access errors
        # Besides that, warmup before the main phase should suffice

        # Restore input structure and relay as actual args.
        n_mem_items = len(mem)
        batch_ref = batches[0]

        def update_aux_single_accel(inputs: 'list[Tensor]') -> 'tuple[Tensor, ...]':
            if n_mem_items:
                inputs, mem = inputs[:-n_mem_items], inputs[-n_mem_items:]

            else:
                mem = []

            input_len = len(inputs) // self.n_truncated_steps
            batches = [batch_ref.from_list(inputs[i:i+input_len]) for i in range(0, len(inputs), input_len)]

            return self.update_aux_single(batches, mem)

        # Capture computational graph
        self.optimizer.zero_grad(set_to_none=True)

        self.update_aux_single_accel, self.accel_graphs['aux'] = capture_graph(
            update_aux_single_accel,
            inputs,
            warmup_tensor_list=(),
            single_input=True)

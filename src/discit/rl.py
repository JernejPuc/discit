"""Reinforcement learning"""

import os
from abc import ABC, abstractmethod
from datetime import timedelta
from io import BytesIO
from time import perf_counter
from typing import Any, Callable

import torch
from torch import cuda, Tensor
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from .accel import capture_graph
from .data import ExperienceBuffer, TensorDict
from .distr import Distribution
from .optim import LRScheduler
from .track import CheckpointTracker


class ActorCritic(Module, ABC):
    def __init__(self):
        Module.__init__(self)

    @abstractmethod
    def init_mem(self, batch_size: int) -> 'tuple[Tensor, ...]':
        ...

    @abstractmethod
    def reset_mem(self, mem: 'tuple[Tensor, ...]', reset_mask: Tensor) -> 'tuple[Tensor, ...]':
        ...

    @abstractmethod
    def get_distr(self, args: 'Tensor | tuple[Tensor, ...]', from_raw: bool) -> Distribution:
        ...

    def unwrap_sample(self, sample: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        return sample[0],

    @abstractmethod
    def act(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        sample: bool
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':
        ...

    @abstractmethod
    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        encode: bool
    ) -> 'tuple[Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]':
        ...

    @abstractmethod
    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        detach: bool
    ) -> 'tuple[Distribution, Distribution, tuple[Tensor, ...]]':
        ...


class PPG:
    """Implementation of phasic policy gradient algorithm focusing on recurrent models."""

    MAX_DISP_SECONDS = 99*24*3600

    def __init__(
        self,
        env_step: Callable[
            ['Tensor | None'],
            'tuple[tuple[Tensor, ...], Tensor, Tensor, dict[str, Any]]'],
        ckpt_tracker: CheckpointTracker,
        scheduler: LRScheduler,
        n_epochs: int,
        log_epoch_interval: int = 1,
        ckpt_epoch_interval: int = 3,
        branch_epoch_interval: int = 10,
        n_rollout_steps: int = 256,
        n_truncated_steps: int = 16,
        batch_size: int = 256,
        n_minibatches: int = 1,
        n_main_iters: int = 8,
        n_aux_iters: int = 6,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_weight: float = 0.5,
        aux_weight: float = 0.,
        entropy_weight: float = 3e-4,
        log_dir: str = 'runs',
        accelerate: bool = True,
        replay_rollout: bool = True,
        update_returns: bool = True,
        detach_critic: bool = True
    ):
        assert ckpt_tracker.model is not None and ckpt_tracker.optimiser is not None

        self.env_step = env_step
        self.ckpter = ckpt_tracker
        self.model: ActorCritic = ckpt_tracker.model
        self.optimiser = ckpt_tracker.optimiser
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
        self.batch_size = batch_size
        self.n_minibatches = n_minibatches
        self.resize_batches = abs(n_minibatches) > 2
        self.shuffle_rng = self.ckpter.rng if self.resize_batches and n_minibatches > 0 else None

        self.n_main_iters = n_main_iters
        self.n_aux_iters = n_aux_iters
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_weight = value_weight
        self.aux_weight = aux_weight
        self.entropy_weight = entropy_weight

        self.main_buffer = ExperienceBuffer(n_rollout_steps)
        self.aux_buffer = ExperienceBuffer(n_rollout_steps * n_main_iters)

        self.score = 0.
        self.reward = torch.tensor(0., device=self.ckpter.device)
        new_zero_tensor = self.reward.clone

        self.lr = 0.
        self.ratio_diff = new_zero_tensor()

        self.stats = {
            'Out/act_mean': new_zero_tensor(),
            'Out/act_std': new_zero_tensor(),
            'Out/val_mean': new_zero_tensor(),
            'Env/reward': new_zero_tensor(),
            'Env/resets': new_zero_tensor(),
            'Main/loss': new_zero_tensor(),
            'Main/policy': new_zero_tensor(),
            'Main/value': new_zero_tensor(),
            'Main/aux': new_zero_tensor(),
            'Main/entropy': new_zero_tensor(),
            'Main/ratio_diff': new_zero_tensor(),
            'GAE/adv_mean': new_zero_tensor(),
            'GAE/adv_std': new_zero_tensor(),
            'Aux/loss': new_zero_tensor(),
            'Aux/value': new_zero_tensor(),
            'Aux/aux': new_zero_tensor(),
            'Aux/kl_div': new_zero_tensor()}

    def run(self):
        """
        Step the model and env. until enough experiences are recorded to update
        model params., repeating the loop for a given number of epochs
        and occasionally making a checkpoint and logging metrics.
        """

        # Initial obs. and mem.
        mem = self.model.init_mem(self.batch_size)
        obs = self.env_step()[0]

        starting_step = epoch_step = self.ckpter.meta['epoch_step']
        starting_time = perf_counter()

        for epoch_step in range(starting_step+1, self.n_epochs+1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time

            remaining_time = min(
                int(running_time * (self.n_epochs - epoch_step + 1) / max(1, epoch_step - 1 - starting_step)),
                self.MAX_DISP_SECONDS)

            # Main phase
            for i in range(1, self.n_main_iters+1):
                self.print_progress(progress, remaining_time, epoch_step, i, True)

                obs, mem = self.collect(obs, mem)

                if self.resize_batches:
                    self.main_buffer = self.main_buffer.restack(self.n_minibatches, self.shuffle_rng)

                updated_mem = self.update_main(i)

                if not self.resize_batches:
                    mem = updated_mem

                self.main_buffer.clear()

            # Aux phase
            if self.resize_batches and not self.update_returns:
                self.aux_buffer = self.aux_buffer.restack(self.n_minibatches, self.shuffle_rng)

            for i in range(1, self.n_aux_iters+1):
                self.print_progress(progress, remaining_time, epoch_step, i, False)

                if self.update_returns:
                    self.recollect(obs, i == 1)

                    if self.resize_batches:
                        self.aux_buffer = self.aux_buffer.restack(self.n_minibatches, self.shuffle_rng)

                updated_mem = self.update_aux()

            if not self.resize_batches:
                mem = updated_mem

            self.aux_buffer.clear()

            # Log running metrics and perf. score
            if self.log_interval and not epoch_step % self.log_interval:
                self.log(epoch_step)

            # Save model params. and training state
            if self.branch_interval and not epoch_step % self.branch_interval:
                self.checkpoint(epoch_step, branch=True)

            elif self.checkpoint_interval and not epoch_step % self.checkpoint_interval:
                self.checkpoint(epoch_step)

        self.checkpoint(epoch_step)
        self.writer.close()

    def print_progress(
        self,
        progress: float,
        remaining_time: float,
        epoch_step: int,
        iter_step: int,
        in_main: bool
    ):
        print(
            f'\rEpoch {epoch_step} of {self.n_epochs} ({progress:.2f}) | '
            f'Iter. {iter_step} of {self.n_main_iters if in_main else self.n_aux_iters} '
            f'({"main" if in_main else "aux"}) | '
            f'ETA: {str(timedelta(seconds=remaining_time))} | '
            f'Score: {self.score:.4f}        ',
            end='')

    def log(self, epoch_step: int):
        n_main = self.n_rollout_steps * self.n_main_iters * self.log_interval
        n_aux = n_main * self.n_aux_iters
        n_gae = self.n_main_iters * self.log_interval
        n_upd = (n_main + n_aux) // self.n_truncated_steps

        env_step = epoch_step * n_main

        for key, val in self.stats.items():
            if key.startswith('Aux'):
                den = n_aux

            elif key.startswith('GAE'):
                den = n_gae

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
                act_out, val_mean, obs_enc, new_mem = self.model.collect(obs, mem, encode=True)

                act = self.model.get_distr(act_out, from_raw=True)
                act_sample = act.sample()

                # Step env.
                obs, rew, rst, *val_aux, info = self.env_step(*self.model.unwrap_sample(act_sample))
                nonrst = 1. - rst

                # Placeholder and aux
                ret = (val_mean, *val_aux)

                # Add batch to buffers
                d = TensorDict({
                    'sample': act_sample,
                    'act': act.args,
                    'val': val_mean,
                    'obs': obs_enc,
                    'mem': mem,
                    'rew': rew,
                    'ret': ret,
                    'rst': rst,
                    'nonrst': nonrst})

                self.main_buffer.append(d)
                self.aux_buffer.append(d)

                # Reset memory if any terminal states are reached
                mem = new_mem

                if torch.any(rst):
                    mem = self.model.reset_mem(mem, rst)

                # Add env. info. to logged metrics
                for key, val in info.items():
                    key = f'Env/{key}'

                    if key not in self.stats:
                        self.stats[key] = torch.tensor(0., device=self.ckpter.device)

                    self.stats[key] += val

            # Perform an additional critic pass to get the final values used in GAE
            _, values, _, _ = self.model.collect(obs, mem, encode=True)

            adv_mean, adv_std = self.main_buffer.label(values, self.discount_factor, self.gae_lambda)

            self.stats['GAE/adv_mean'] += adv_mean
            self.stats['GAE/adv_std'] += adv_std

        return obs, mem

    def recollect(self, final_obs: 'tuple[Tensor, ...]', update_act: bool):
        mem = self.aux_buffer.batches[0]['mem']

        with torch.no_grad():
            for b in self.aux_buffer.batches:

                # Step actor
                act_out, val_mean, _, new_mem = self.model.collect(b['obs'], mem, encode=False)

                # Update batch
                if update_act:
                    b['act'] = self.model.get_distr(act_out, from_raw=True).args

                b['val'] = val_mean
                b['mem'] = mem

                # Reset memory if any terminal states are reached
                mem = new_mem

                if torch.any(b['rst']):
                    mem = self.model.reset_mem(mem, b['rst'])

            # Perform an additional critic pass to get the final values used in GAE
            _, values, _, _ = self.model.collect(final_obs, mem, encode=True)

            # Update target returns
            self.aux_buffer.label(values, self.discount_factor, self.gae_lambda, skip_std=True)

    def update_main(self, iter_num: int) -> 'tuple[Tensor, ...]':
        """
        Iterate over sequences of batches in a rollout and update
        model params. according to the main objective with epochwise TBPTT.
        """

        self.reward.zero_()
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
                self.optimiser.zero_grad()
                mem = self.update_main_single(batches, mem)

            self.scheduler.step(self.ratio_diff.item() / self.n_truncated_steps)
            self.lr += self.scheduler.lr

        # Update print-out info
        score = self.stats.get('Env/score')

        if score is None:
            self.score = self.reward.item() / self.n_rollout_steps

        else:
            self.score = score.item() / (self.n_rollout_steps * iter_num)

        return tuple([m.detach() for m in mem])

    def update_main_single(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        stats = self.stats
        running_loss = 0.

        for batch in batches:
            act, val, *aux, mem = self.model(batch['obs'], mem, detach=self.detach_critic)
            ret, *ref_aux = batch['ret']

            act: Distribution
            val: Distribution
            aux: 'tuple[Distribution]'

            mem = self.model.reset_mem(mem, batch['rst'])

            # Policy
            old_act: Distribution = self.model.get_distr(batch['act'], from_raw=False)

            act_log_prob = act.log_prob(*batch['sample'])
            old_act_log_prob = old_act.log_prob(*batch['sample'])

            # Bound ratio to [0.05, 20] for stability
            old_act_log_prob = old_act_log_prob.clamp(act_log_prob-3., act_log_prob+3.)

            ratio = (act_log_prob - old_act_log_prob).exp()

            policy_loss = -torch.minimum(
                batch['adv'] * ratio,
                batch['adv'] * ratio.clamp(1. - self.clip_ratio, 1. + self.clip_ratio)).mean()

            # Value
            # NOTE: No value loss clipping
            value_loss = -val.log_prob(ret.unsqueeze(-1)).mean()

            # Auxiliary
            aux_loss = 0.

            for aux_i, ref_aux_i in zip(aux, ref_aux):
                aux_loss = aux_loss - aux_i.log_prob(ref_aux_i).mean()

            # Entropy
            entropy = act.entropy.mean()

            # Total
            full_loss = (
                policy_loss
                + self.value_weight * value_loss
                + self.aux_weight * aux_loss
                - self.entropy_weight * entropy)

            running_loss = running_loss + full_loss

            # Stats for logging
            with torch.no_grad():
                reward = batch['rew'].mean()
                ratio_diff = (ratio - 1.).abs().mean()

                self.reward += reward
                self.ratio_diff += ratio_diff

                stats['Out/act_mean'] += act.mean.mean()
                stats['Out/act_std'] += act.dev.mean()
                stats['Out/val_mean'] += batch['val'].mean()
                stats['Main/loss'] += full_loss
                stats['Main/policy'] += policy_loss
                stats['Main/value'] += value_loss
                stats['Main/aux'] += aux_loss
                stats['Main/entropy'] += entropy
                stats['Main/ratio_diff'] += ratio_diff
                stats['Env/reward'] += reward
                stats['Env/resets'] += batch['rst'].sum()

        # Average loss over N time steps for TBPTT
        # NOTE: https://r2rt.com/styles-of-truncated-backpropagation.html
        loss = running_loss / self.n_truncated_steps
        loss.backward()

        self.optimiser.step()

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
            torch.save(self.optimiser.state_dict(), optim_state_bytes)

            model_state_bytes.seek(0)
            optim_state_bytes.seek(0)

            # Warmup steps
            for _ in range(3):
                self.optimiser.zero_grad(set_to_none=True)
                self.update_main_single(batches, mem)

            # Restore state before warmup
            self.reward.fill_(reward)

            for v, v_ in zip(self.stats.values(), stats_values):
                v.fill_(v_)

            self.model.load_state_dict(torch.load(model_state_bytes))
            self.optimiser.load_state_dict(torch.load(optim_state_bytes))

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
        self.optimiser.zero_grad(set_to_none=True)

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

        self.reward.zero_()
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
                self.optimiser.zero_grad()
                mem = self.update_aux_single(batches, mem)

            self.scheduler.step()
            self.lr += self.scheduler.lr

        # Update print-out info
        self.score = self.stats.get('Env/score', self.reward).item() / (self.n_rollout_steps * self.n_main_iters)

        return tuple([m.detach() for m in mem])

    def update_aux_single(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        stats = self.stats
        running_loss = 0.

        for batch in batches:
            act, val, *aux, mem = self.model(batch['obs'], mem, detach=False)
            ret, *ref_aux = batch['ret']

            act: Distribution
            val: Distribution
            aux: 'tuple[Distribution]'

            mem = self.model.reset_mem(mem, batch['rst'])

            # KL divergence
            old_act: Distribution = self.model.get_distr(batch['act'], from_raw=False)

            kl_div = old_act.kl_div(act).mean()

            # Value
            # NOTE: No value loss clipping
            value_loss = -val.log_prob(ret.unsqueeze(-1)).mean()

            # Auxiliary
            aux_loss = 0.

            for aux_i, ref_aux_i in zip(aux, ref_aux):
                aux_loss = aux_loss - aux_i.log_prob(ref_aux_i).mean()

            # Total
            full_loss = value_loss + self.aux_weight * aux_loss + kl_div
            running_loss = running_loss + full_loss

            # Stats for logging
            with torch.no_grad():
                reward = batch['rew'].mean()
                self.reward += reward

                stats['Aux/loss'] += full_loss
                stats['Aux/value'] += value_loss
                stats['Aux/aux'] += aux_loss
                stats['Aux/kl_div'] += kl_div

        # Average loss over N time steps for TBPTT
        # NOTE: https://r2rt.com/styles-of-truncated-backpropagation.html
        loss = running_loss / self.n_truncated_steps
        loss.backward()

        self.optimiser.step()

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
        self.optimiser.zero_grad(set_to_none=True)

        self.update_aux_single_accel, self.accel_graphs['aux'] = capture_graph(
            update_aux_single_accel,
            inputs,
            warmup_tensor_list=(),
            single_input=True)

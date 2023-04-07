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
from .distr import ActionDistrTemplate, ValueDistrTemplate
from .optim import SoftConstLRScheduler
from .track import CheckpointTracker


class ActorCriticTemplate(Module, ABC):
    MODE_LEARNER = 0
    MODE_COLLECTOR = 1
    MODE_ACTOR = 2
    MODE_CRITIC = 3

    def __init__(self):
        Module.__init__(self)

        self.fwd_map = (self.fwd_learner, self.fwd_collector, self.fwd_actor, self.fwd_critic)

    @abstractmethod
    def init_mem(self, batch_size: int, detach: bool) -> 'tuple[Tensor, ...]':
        raise NotImplementedError

    @abstractmethod
    def reset_mem(self, mem: 'tuple[Tensor, ...]', reset_mask: Tensor, keep_mask: Tensor) -> 'tuple[Tensor, ...]':
        raise NotImplementedError

    @abstractmethod
    def get_distr(self, args: 'tuple[Tensor, ...]') -> ActionDistrTemplate:
        raise NotImplementedError

    @abstractmethod
    def fwd_actor(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':
        raise NotImplementedError

    @abstractmethod
    def fwd_critic(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':
        raise NotImplementedError

    @abstractmethod
    def fwd_collector(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[tuple[Tensor, ...], ActionDistrTemplate, Tensor, Tensor, tuple[Tensor, ...]]':
        raise NotImplementedError

    @abstractmethod
    def fwd_learner(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[ActionDistrTemplate, ValueDistrTemplate, tuple[Tensor, ...]]':
        raise NotImplementedError

    def forward(self, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]', mode: int = MODE_LEARNER):
        return self.fwd_map[mode](obs, mem)


class PPG:
    MAX_DISP_SECONDS = 99*24*3600

    def __init__(
        self,
        env_step: Callable[
            ['Tensor | None'],
            'tuple[tuple[Tensor, ...], Tensor, Tensor, dict[str, Any]]'],
        ckpt_tracker: CheckpointTracker,
        scheduler: SoftConstLRScheduler,
        n_epochs: int,
        log_epoch_interval: int = 1,
        ckpt_epoch_interval: int = 3,
        branch_epoch_interval: int = 10,
        n_rollout_steps: int = 256,
        n_truncated_steps: int = 16,
        batch_size: int = 256,
        n_main_iters: int = 8,
        n_aux_iters: int = 6,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_weight: float = 0.5,
        entropy_weight: float = 3e-4,
        log_dir: str = 'runs',
        accelerate: bool = True,
        replay_rollout: bool = True
    ):
        assert ckpt_tracker.model is not None and ckpt_tracker.optimiser is not None

        self.env_step = env_step
        self.ckpter = ckpt_tracker
        self.model: ActorCriticTemplate = ckpt_tracker.model
        self.optimiser = ckpt_tracker.optimiser
        self.scheduler = scheduler

        self.accelerate = accelerate
        self.accel_graphs = {}
        self.update_main_single_accel = None
        self.update_aux_single_accel = None

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, ckpt_tracker.model_name))
        self.write = self.writer.add_scalar

        self.n_epochs = n_epochs
        self.log_interval = log_epoch_interval
        self.checkpoint_interval = ckpt_epoch_interval
        self.branch_interval = branch_epoch_interval

        self.replay_rollout = replay_rollout
        self.n_rollout_steps = n_rollout_steps
        self.n_truncated_steps = n_truncated_steps
        self.batch_size = batch_size

        self.n_main_iters = n_main_iters
        self.n_aux_iters = n_aux_iters
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

        self.main_buffer = ExperienceBuffer(n_rollout_steps)
        self.aux_buffer = ExperienceBuffer(n_rollout_steps * n_main_iters)

        self.score = 0.
        self.reward = torch.tensor(0., device=self.ckpter.device)
        new_zero_tensor = self.reward.clone

        self.stats = {
            'col_act_mean': new_zero_tensor(),
            'col_act_std': new_zero_tensor(),
            'col_val_mean': new_zero_tensor(),
            'main_loss': new_zero_tensor(),
            'main_policy': new_zero_tensor(),
            'main_value': new_zero_tensor(),
            'main_entropy': new_zero_tensor(),
            'main_ratio_diff': new_zero_tensor(),
            'main_adv_mean': new_zero_tensor(),
            'main_adv_std': new_zero_tensor(),
            'aux_loss': new_zero_tensor(),
            'aux_value': new_zero_tensor(),
            'aux_kl_div': new_zero_tensor(),
            'env_reward': new_zero_tensor(),
            'env_resets': new_zero_tensor()}

    def run(self):
        """
        Step the model and env. until enough experiences are recorded to update
        model params., repeating the loop for a given number of epochs
        and occasionally making a checkpoint and logging metrics.
        """

        # Initial obs. and mem.
        mem = self.model.init_mem(self.batch_size, detach=True)
        obs = self.env_step()[0]

        starting_step = epoch_step = self.ckpter.meta['epoch_step']
        starting_time = perf_counter()

        for epoch_step in range(starting_step+1, self.n_epochs+1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time

            remaining_time = min(
                int(running_time * (self.n_epochs - epoch_step + 1) / (epoch_step - starting_step)),
                self.MAX_DISP_SECONDS)

            # Main phase
            for i in range(1, self.n_main_iters+1):
                self.print_progress(progress, remaining_time, epoch_step, i, True)

                obs, mem = self.collect(obs, mem)

                mem = self.update_main()

                self.main_buffer.clear()

            # Aux phase
            for i in range(1, self.n_aux_iters+1):
                self.print_progress(progress, remaining_time, epoch_step, i, False)

                mem = self.update_aux()

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
        den_main = self.n_rollout_steps * self.n_main_iters
        den_aux = den_main * max(1, self.n_aux_iters)

        # To keep loop simple
        self.stats['main_adv_mean'] *= den_main / self.n_main_iters
        self.stats['main_adv_std'] *= den_main / self.n_main_iters

        den_main *= self.log_interval
        den_aux *= self.log_interval

        for key, val in tuple(self.stats.items()):
            self.write(key, val.item() / (den_aux if key.startswith('aux') else den_main), epoch_step)
            self.stats[key].zero_()

    def checkpoint(self, epoch_step: int, branch: bool = False):
        update_step = self.scheduler.step_ctr
        ckpt_increment = 1 if branch else 0

        self.ckpter.checkpoint(epoch_step, update_step, ckpt_increment, self.score)

    def collect(self, obs: Tensor, mem: Tensor):
        with torch.no_grad():
            for _ in range(self.n_rollout_steps):

                # Step actor
                act, val, obs_enc, new_mem = self.model.fwd_collector(obs, mem)
                act_sample = self.model.get_distr(act).sample

                # Step env.
                obs, rew, rst, info = self.env_step(act_sample)
                nrst = 1. - rst

                # Add batch to buffers
                d = TensorDict({
                    'act': act,
                    'val': val,
                    'obs': obs_enc,
                    'mem': mem,
                    'rew': rew,
                    'rst': rst,
                    'nrst': nrst})

                self.main_buffer.append(d)
                self.aux_buffer.append(d)

                # Reset memory if any terminal states are reached
                mem = new_mem

                if torch.any(rst):
                    mem = self.model.reset_mem(mem, rst, nrst)

                # Add env. info. to logged metrics
                for key, val in info.items():
                    key = f'env_{key}'

                    if key not in self.stats:
                        self.stats[key] = torch.tensor(0., device=self.ckpter.device)

                    self.stats[key] += val

            # Perform an additional critic pass to get the final values used in GAE
            values, _ = self.model.fwd_critic(obs, mem)

            adv_mean, adv_std = self.main_buffer.label(values, self.discount_factor, self.gae_lambda)

            self.stats['main_adv_mean'] += adv_mean
            self.stats['main_adv_std'] += adv_std

        return obs, mem

    def update_main(self) -> 'tuple[Tensor, ...]':
        """
        Iterate over sequences of batches in a rollout and update
        model params. according to the main objective with epochwise TBPTT.
        """

        self.reward.zero_()
        mem = None

        for batches in self.main_buffer.iter_slices(self.n_truncated_steps):

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
                if self.update_main_single_accel is None:
                    self.accel_main(batches, mem, full_input_list)

                mem = tuple(self.update_main_single_accel(full_input_list))

            else:
                self.optimiser.zero_grad()
                mem = self.update_main_single(batches, mem)

            self.scheduler.step()

        # Update print-out info
        self.score = self.stats.get('env_score', self.reward).item() / self.n_rollout_steps

        return tuple([m.detach() for m in mem])

    def update_main_single(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        stats = self.stats
        running_loss = 0.

        for batch in batches:
            act, val, mem = self.model(batch['obs'], mem)
            act: ActionDistrTemplate
            val: ValueDistrTemplate

            mem = self.model.reset_mem(mem, batch['rst'], batch['nrst'])

            # Policy
            old_act: ActionDistrTemplate = self.model.get_distr(batch['act'])
            act_sample = old_act.sample

            ratio = (act.log_prob(act_sample) - old_act.log_prob(act_sample)).exp()

            policy_loss = -torch.minimum(
                batch['adv'] * ratio,
                batch['adv'] * ratio.clamp(1. - self.clip_ratio, 1. + self.clip_ratio)).mean()

            # Value
            # NOTE: No value loss clipping
            value_loss = -val.log_prob(batch['ret']).mean()

            # Entropy
            entropy = act.entropy.mean()

            # Total
            full_loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
            running_loss = running_loss + full_loss

            # Stats for logging
            with torch.no_grad():
                reward = batch['rew'].mean()
                self.reward += reward

                stats['col_act_mean'] += act.loc.mean()
                stats['col_act_std'] += act.scale.mean()
                stats['col_val_mean'] += batch['val'].mean()
                stats['main_loss'] += full_loss
                stats['main_policy'] += policy_loss
                stats['main_value'] += value_loss
                stats['main_entropy'] += entropy
                stats['main_ratio_diff'] += (ratio - 1.).abs().mean()
                stats['env_reward'] += reward
                stats['env_resets'] += batch['rst'].sum()

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
            mem = [inputs.pop() for _ in range(n_mem_items)][::-1]

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

        # Update print-out info
        self.score = self.stats.get('env_score', self.reward).item() / (self.n_rollout_steps * self.n_main_iters)

        return tuple([m.detach() for m in mem])

    def update_aux_single(self, batches: 'list[TensorDict]', mem: 'tuple[Tensor, ...]') -> 'tuple[Tensor, ...]':
        stats = self.stats
        running_loss = 0.

        for batch in batches:
            act, val, mem = self.model(batch['obs'], mem)
            act: ActionDistrTemplate
            val: ValueDistrTemplate

            mem = self.model.reset_mem(mem, batch['rst'], batch['nrst'])

            # KL divergence
            old_act: ActionDistrTemplate = self.model.get_distr(batch['act'])

            kl_div = old_act.kl_div(act).mean()

            # Value
            # NOTE: No value loss clipping
            value_loss = -val.log_prob(batch['ret']).mean()

            # Total
            full_loss = value_loss + kl_div
            running_loss = running_loss + full_loss

            # Stats for logging
            with torch.no_grad():
                reward = batch['rew'].mean()
                self.reward += reward

                stats['aux_loss'] += full_loss
                stats['aux_value'] += value_loss
                stats['aux_kl_div'] += kl_div

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
            mem = [inputs.pop() for _ in range(n_mem_items)][::-1]

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

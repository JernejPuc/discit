"""Single-agent reinforcement learning"""

from typing import Callable

import torch
from torch import Tensor

from .accel import capture_graph
from .data import ExperienceBuffer, TensorDict
from .distr import Distribution
from .marl import AuxTask, MAXPPO, MultiActorCritic as ActorCritic
from .optim import LRScheduler
from .track import CheckpointTracker


class PPO(MAXPPO):
    """Proximal policy optimisation"""

    def __init__(
        self,
        env_step: Callable[
            ['Tensor | None', 'Tensor | None'],
            'tuple[tuple[Tensor, ...], dict[str, Tensor | tuple[Tensor, ...]], dict[str, float]]'],
        ckpt_tracker: CheckpointTracker,
        scheduler: LRScheduler,
        n_actors: int,
        n_epochs: int,
        log_epoch_interval: int = 1,
        ckpt_epoch_interval: int = 1,
        branch_epoch_interval: int = 0,
        n_rollout_steps: int = 1,
        n_truncated_steps: int = 1,
        n_passes_per_step: int = 1,
        n_passes_per_rollout: 'int | float' = 1,
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
        n_envs = n_actors
        buffer_size = int(n_rollout_steps * n_passes_per_rollout)

        super().__init__(
            env_step, ckpt_tracker, scheduler, n_envs, n_actors, n_epochs,
            log_epoch_interval, ckpt_epoch_interval, branch_epoch_interval,
            n_rollout_steps, n_truncated_steps, n_passes_per_step,
            buffer_size, batch_size, discount_gammas, trace_lambda, clip_ratio,
            policy_weight, value_weight, aux_weight, entropy_weight,
            aux_task, log_dir, bias_returns, accelerate)


class PPGAuxTask(AuxTask):
    STAT_KEYS = ('Aux/value', 'Aux/val_aux', 'Aux/kl_div')

    def __init__(
        self,
        ckpt_tracker: CheckpointTracker,
        n_rollout_steps: int,
        n_truncated_steps: int,
        n_passes_per_step: int,
        n_main_iters: int,
        n_aux_iters: int,
        discount_gammas: 'float | tuple[float, ...]',
        trace_lambda: float,
        value_weight: float,
        aux_weight: float,
        kl_weight: float,
        bias_returns: bool,
        accelerate: bool = False
    ):
        super().__init__(offline=True)

        self.model: ActorCritic = ckpt_tracker.model
        self.optimizer = ckpt_tracker.optimizer
        self.rng = ckpt_tracker.rng

        self.accelerate = accelerate
        self.accel_graph = None
        self.update_single_accel = None

        self.n_rollout_steps = n_rollout_steps
        self.n_truncated_steps = n_truncated_steps
        self.n_aux_iters = n_aux_iters

        # Aux. update is called after each main update
        # A counter is used to defer aux. update until the final main update in an epoch
        self.n_main_updates = n_rollout_steps // n_truncated_steps * n_passes_per_step * n_main_iters
        self.main_update_ctr = 0

        if hasattr(discount_gammas, '__len__'):
            discount_gammas = torch.tensor((discount_gammas,), dtype=torch.float32, device=ckpt_tracker.device)

        self.discount_gammas = discount_gammas
        self.trace_lambda = trace_lambda
        self.value_weight = value_weight
        self.aux_weight = aux_weight
        self.kl_weight = kl_weight
        self.bias_returns = bias_returns

        self.aux_buffer = ExperienceBuffer(n_rollout_steps * n_main_iters)

        self.final_inputs: 'tuple[tuple[Tensor, ...], tuple[Tensor, ...], None]' = None
        self.stats: dict[str, Tensor] = {k: 0. for k in self.STAT_KEYS}

    def clear(self):
        self.aux_buffer.clear(self.n_rollout_steps)

    def collect(self, data: TensorDict, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):
        self.aux_buffer.append(data)
        self.final_inputs = obs, mem, None

    def recollect(self, update_act: bool):
        mem = self.aux_buffer.batches[0]['mem']

        for b in self.aux_buffer.batches:

            # Step actors
            data, _, mem = self.model.collect(b['obs'], mem, b['act'])

            # Update batch
            if update_act:
                b['act'] = data['args']

            b['val'] = data['val']
            b['mem'] = data['mem']

            # Reset memory if any terminal states are reached
            if not b['nrst'].all():
                mem = self.model.reset_mem(mem, b['nrst'])

        # Perform an additional critic pass to get the final values used in GAE
        values = self.model.collect(*self.final_inputs)[0]['val']

        # Update return targets
        self.aux_buffer.label(
            values, self.discount_gammas, self.trace_lambda, bias_returns=self.bias_returns, skip_std=True)

    def update(self, batches: 'list[TensorDict]', stats: 'dict[str, Tensor]'):
        if not self.aux_buffer.is_full():
            return

        self.main_update_ctr += 1

        if self.main_update_ctr < self.n_main_updates:
            return

        self.main_update_ctr = 0
        self.stats = stats

        for i in range(self.n_aux_iters):
            with torch.no_grad():
                self.recollect(update_act=i == 0)

            buffer = self.aux_buffer.shuffle(self.rng, seq_length=self.n_truncated_steps)

            for seq in buffer.iter_slices(self.n_truncated_steps):

                # CUDA graph acceleration
                if self.accelerate:

                    # Flatten content of batches into a list of tensors to pass to the graph
                    full_input_list = [t for b in seq for t in b.to_list()]

                    # Capture computational graph
                    if self.accel_graph is None:
                        self.accel_update(seq, full_input_list)

                    self.update_single_accel(full_input_list)

                else:
                    self.update_single(seq)

                self.n_update_steps += self.n_truncated_steps

    def update_single(self, batches: 'list[TensorDict]'):
        stats = self.stats
        running_loss = 0.
        mem = batches[0]['mem']

        self.optimizer.zero_grad()

        for batch in batches:
            data = self.model(batch['obs'], mem, batch['act'], detach=False)
            mem = self.model.reset_mem(data['mem'], batch['nrst'])

            running_loss = running_loss + self.loss(batch, data['act'], (data['val'],), data['aux'], stats)

        # Average loss over N time steps for TBPTT
        # NOTE: https://r2rt.com/styles-of-truncated-backpropagation.html
        running_loss = running_loss / self.n_truncated_steps
        running_loss.backward()

        self.optimizer.step()

    def loss(
        self,
        batch: TensorDict,
        act: Distribution,
        vals: 'tuple[Distribution, ...]',
        auxs: 'tuple[Distribution, ...]',
        stats: 'dict[str, Tensor]'
    ) -> Tensor:

        # KL divergence
        old_act: Distribution = self.model.get_distr(batch['args'])

        kl_div = old_act.kl_div(act).mean()

        # Value
        value_loss = vals[0].log_prob(batch['ret']).mean()

        # Auxiliary
        aux_loss = 0.

        for aux in auxs:
            aux_loss = aux_loss + aux.log_prob(batch['ret']).mean()

        # Total
        loss = (
            self.kl_weight * kl_div
            - self.value_weight * value_loss
            - self.aux_weight * aux_loss)

        # Stats for logging
        with torch.no_grad():
            stats['Aux/loss'] += loss
            stats['Aux/value'] -= value_loss
            stats['Aux/val_aux'] -= aux_loss
            stats['Aux/kl_div'] += kl_div

        return loss

    def accel_update(self, batches: 'list[TensorDict]', inputs: 'list[Tensor]'):

        # NOTE: Repeating warmup here resulted in illegal CUDA memory access errors
        # Besides that, warmup before the main phase should suffice

        # Restore input structure and relay as actual args.
        batch_ref = batches[0]

        def update_single_accel(inputs: 'list[Tensor]'):
            input_len = len(inputs) // self.n_truncated_steps
            batches = [batch_ref.from_list(inputs[i:i+input_len]) for i in range(0, len(inputs), input_len)]

            self.update_single(batches)

        # Capture computational graph
        self.update_single_accel, self.accel_graph = capture_graph(
            update_single_accel,
            inputs,
            warmup_tensor_list=(),
            single_input=True)


class PPG(MAXPPO):
    """Phasic policy gradients"""

    def __init__(
        self,
        env_step: Callable[
            ['Tensor | None', 'Tensor | None'],
            'tuple[tuple[Tensor, ...], dict[str, Tensor | tuple[Tensor, ...]], dict[str, float]]'],
        ckpt_tracker: CheckpointTracker,
        scheduler: LRScheduler,
        n_actors: int,
        n_epochs: int,
        log_epoch_interval: int = 1,
        ckpt_epoch_interval: int = 1,
        branch_epoch_interval: int = 0,
        n_rollout_steps: int = 1,
        n_truncated_steps: int = 1,
        n_passes_per_step: int = 1,
        n_passes_per_rollout: 'int | float' = 1,
        n_main_iters: int = 1,
        n_aux_iters: int = 1,
        batch_size: int = None,
        discount_gammas: 'float | tuple[float, ...]' = 0.99,
        trace_lambda: float = 0.95,
        clip_ratio: float = 0.25,
        policy_weight: float = 1.,
        value_weight: float = 0.5,
        aux_weight: float = 0.5,
        kl_weight: float = 1.,
        entropy_weight: 'float | Tensor' = 1e-3,
        log_dir: str = 'runs',
        bias_returns: bool = False,
        accelerate: bool = False
    ):
        n_envs = n_actors
        buffer_size = int(n_rollout_steps * n_passes_per_rollout)

        aux_task = PPGAuxTask(
            ckpt_tracker,
            buffer_size, n_truncated_steps, n_passes_per_step,
            n_main_iters, n_aux_iters, discount_gammas, trace_lambda,
            value_weight, aux_weight, kl_weight,
            bias_returns, accelerate)

        super().__init__(
            env_step, ckpt_tracker, scheduler, n_envs, n_actors, n_epochs,
            log_epoch_interval, ckpt_epoch_interval, branch_epoch_interval,
            n_rollout_steps, n_truncated_steps, n_passes_per_step,
            buffer_size, batch_size, discount_gammas, trace_lambda, clip_ratio,
            policy_weight, value_weight, aux_weight, entropy_weight,
            aux_task, log_dir, bias_returns, accelerate)

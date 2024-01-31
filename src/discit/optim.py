"""NN optimisation"""

from math import cos, pi
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class NAdamW(Optimizer):
    """Adam with Nesterov momentum and decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: 'tuple[float, float]' = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 1e-4,
        clip_grad_value: float = 4.,
        device: 'str | torch.device' = 'cuda'
    ):
        defaults = dict(
            lr=torch.tensor(lr, dtype=torch.float32, device=device),
            beta1=torch.tensor(betas[0], dtype=torch.float32, device=device),
            beta1_next=torch.tensor(betas[0], dtype=torch.float32, device=device),
            beta2=torch.tensor(betas[1], dtype=torch.float32, device=device),
            beta1_product=torch.tensor(1., dtype=torch.float32, device=device),
            beta2_product=torch.tensor(1., dtype=torch.float32, device=device),
            eps=torch.tensor(eps, dtype=torch.float32, device=device),
            weight_decay=(torch.tensor(weight_decay, dtype=torch.float32, device=device) if weight_decay else None),
            clip_grad_value=(clip_grad_value if clip_grad_value else None),
            step=torch.tensor(0, dtype=torch.int64, device=device))

        super().__init__(params, defaults)

        # Init EWMA of gradients and squared gradients
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if not p.requires_grad:
                        continue

                    state: 'dict[str, Tensor]' = self.state[p]

                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:

            # Unpack
            neg_lr = -group['lr']
            beta1 = group['beta1']
            beta1_next = group['beta1_next']
            beta2 = group['beta2']
            beta1_product = group['beta1_product']
            beta2_product = group['beta2_product']
            eps = group['eps']
            weight_mul = (1. + neg_lr * group['weight_decay']) if group['weight_decay'] is not None else None
            clip_grad_value = group['clip_grad_value'] if group['clip_grad_value'] is not None else None
            neg_clip_grad_value = -clip_grad_value if clip_grad_value is not None else None

            # Update
            group['step'] += 1
            beta2_product *= beta2
            beta1_product *= beta1
            beta1_product_next = beta1_product * beta1_next

            one_minus_beta1 = 1. - beta1
            one_minus_beta2 = 1. - beta2

            bias_correction2 = 1. - beta2_product
            bias_correction1 = 1. - beta1_product
            bias_correction1_next = 1. - beta1_product_next

            grad_step = neg_lr * one_minus_beta1 / bias_correction1
            momentum_step = neg_lr * beta1_next / bias_correction1_next

            # Apply per param with valid grad
            for p in group['params']:
                if p.grad is None:
                    continue

                # Unpack EWMA of gradients and squared gradients
                state: 'dict[str, Tensor]' = self.state[p]

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                param: Tensor = p
                grad: Tensor = p.grad

                if clip_grad_value is not None:
                    grad.data.nan_to_num_(0.)
                    grad.data.clip_(neg_clip_grad_value, clip_grad_value)

                # Decoupled weight decay
                if weight_mul is not None:
                    param.mul_(weight_mul)

                # Update EWMA of gradients and squared gradients
                exp_avg.mul_(beta1).add_(grad * one_minus_beta1)                # alpha
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad * one_minus_beta2)   # value

                denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)

                # Update param
                param.addcdiv_(grad * grad_step, denom)                         # value
                param.addcdiv_(exp_avg * momentum_step, denom)                  # value


class LRScheduler:
    """Constant learning rate scheduler that only increments the step counter."""

    lrs_init: 'list[float]'
    lrs: 'list[float]'
    lr: float
    step_ctr: int

    def __init__(self, optimizer: Optimizer, starting_step: int = 0):
        self.optimizer = optimizer
        self.lrs_init = [float(group['lr']) for group in optimizer.param_groups]
        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.lrs = self.lrs_init
        self.lr = sum(self.lrs) / len(self.lrs)

    def step(self, _metrics: Any = None, increment: int = 1):
        self.step_ctr += increment

    def update(self, scheduler: 'LRScheduler'):
        self.lrs = scheduler.lrs
        self.lr = scheduler.lr

    def update_args(self, **shared_args: 'dict[str, float]'):
        with torch.no_grad():
            for lr, param_group in zip(self.lrs, self.optimizer.param_groups):
                param_group['lr'].fill_(lr)

                if shared_args:
                    for k, v in shared_args.items():
                        param_group[k].fill_(v)

    def state_dict(self) -> 'dict[str, Any]':
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict: 'dict[str, Any]'):
        self.__dict__.update(state_dict)


class AnnealingScheduler(LRScheduler):
    """
    Tri-phase annealing learning rate scheduler, separating warmup and cooldown
    with a constant middle (main) section for learning at the maximum rate.

    When the learning rate is changed by another scheduler, the max. LR attrs.
    are the ones being changed.
    """

    in_main: bool

    def __init__(
        self,
        optimizer: Optimizer,
        step_milestones: 'tuple[int, int, int]',
        lr_div_factors: 'tuple[float, float]' = (20., 400.),
        beta1_bounds: 'tuple[float, float]' = (0.85, 0.95),
        cosine: bool = True,
        starting_step: int = 0
    ):
        self.step_start_main, self.step_end_main, self.step_total = step_milestones
        lr_div_init, lr_div_final = lr_div_factors
        self.beta1_min, self.beta1_max = beta1_bounds
        self.cosine = cosine

        self.lrs_max = [float(group['lr']) for group in optimizer.param_groups]
        self.lrs_init = [lr / lr_div_init for lr in self.lrs_max]
        self.lrs_final = [lr / lr_div_final for lr in self.lrs_max]

        self.optimizer = optimizer
        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        LRScheduler.reset(self, starting_step)
        self.in_main = False
        self.update_args(**self.anneal())

    def step(self, _value: float = None, increment: int = 1):
        self.step_ctr += increment

        # Keep constant in main phase
        if self.in_main:
            if self.step_ctr < self.step_end_main:
                return

            self.in_main = False

        self.update_args(**self.anneal())

    def update(self, scheduler: LRScheduler):
        self.lrs_max = [lr_max * (lr / new_lr) for lr_max, lr, new_lr in zip(self.lrs_max, self.lrs, scheduler.lrs)]

        LRScheduler.update(self, scheduler)

    def anneal(self) -> 'dict[str, float]':
        if self.step_ctr < self.step_start_main:
            ratio = self.get_ratio(self.step_ctr, self.step_start_main, self.cosine)
            ratio_next = self.get_ratio(self.step_ctr + 1, self.step_start_main, self.cosine)

            self.lrs = [self.lerp(lr_init, lr_max, ratio) for lr_init, lr_max in zip(self.lrs_init, self.lrs_max)]
            beta1 = self.lerp(self.beta1_max, self.beta1_min, ratio)
            beta1_next = self.lerp(self.beta1_max, self.beta1_min, ratio_next)

        elif self.step_ctr >= self.step_end_main:
            shifted_ctr = self.step_ctr - self.step_end_main
            shifted_total = self.step_total - self.step_end_main

            ratio = self.get_ratio(shifted_ctr, shifted_total, self.cosine)
            ratio_next = self.get_ratio(shifted_ctr + 1, shifted_total, self.cosine)

            self.lrs = [self.lerp(lr_max, lr_final, ratio) for lr_max, lr_final in zip(self.lrs_max, self.lrs_final)]
            beta1 = self.lerp(self.beta1_min, self.beta1_max, ratio)
            beta1_next = self.lerp(self.beta1_min, self.beta1_max, ratio_next)

        else:
            self.in_main = True

            self.lrs = self.lrs_max
            beta1 = self.beta1_min
            beta1_next = self.beta1_min

        self.lr = sum(self.lrs) / len(self.lrs)

        return {'beta1': beta1, 'beta1_next': beta1_next}

    @staticmethod
    def get_ratio(num: int, den: int, cosine: bool) -> float:
        ratio = max(0., min(1., num / max(1, den)))

        if cosine:
            ratio = (1. - cos(pi * ratio)) / 2.

        return ratio

    @staticmethod
    def lerp(start: float, end: float, ratio: float) -> float:
        return start + (end - start) * ratio


class PlateauReducingScheduler(LRScheduler):
    """Reduces LR on plateau, i.e. when a metric has stopped improving."""

    best_value: float
    waiting_steps: int

    def __init__(
        self,
        optimizer: Optimizer,
        lr_min: float = 1e-6,
        scale_factor: float = 0.7,
        max_waiting_steps: int = 10,
        expect_increase: bool = True,
        starting_step: int = 0
    ):
        self.lr_min = lr_min
        self.down_scale = scale_factor
        self.max_waiting_steps = max_waiting_steps
        self.expect_increase = expect_increase

        LRScheduler.__init__(self, optimizer, starting_step)

    def reset(self, starting_step: int = 0):
        LRScheduler.reset(self, starting_step)
        self.best_value = float('-inf' if self.expect_increase else 'inf')
        self.waiting_steps = 0
        self.update_args()

    def step(self, value: float = None, increment: int = 1):
        self.step_ctr += increment

        if value is None:
            return

        if value > self.best_value if self.expect_increase else value < self.best_value:
            self.best_value = value
            self.waiting_steps = 0

        else:
            self.waiting_steps += 1

        if self.waiting_steps <= self.max_waiting_steps:
            return

        self.lrs = [max(self.lr_min, lr * self.down_scale) for lr in self.lrs]
        self.lr = sum(self.lrs) / len(self.lrs)

        self.waiting_steps = 0
        self.update_args()

    def update(self, scheduler: LRScheduler):
        if scheduler.lr < self.lr:
            self.waiting_steps = 0

        LRScheduler.update(self, scheduler)


class BoundingScheduler(LRScheduler):
    """
    Decreases the learning rate if the tracked value average exceeds
    the given thresholds.
    """

    window: 'list[float]'
    window_ptr: int

    def __init__(
        self,
        optimizer: Optimizer,
        lr_min: float = 1e-7,
        val_refs: 'tuple[float, float, float]' = (0.1, 0., 0.2),
        scale_factor: float = 0.7,
        window_len: int = 10,
        starting_step: int = 0
    ):
        self.lr_min = lr_min
        self.val_init, self.val_min, self.val_max = val_refs
        self.down_scale = scale_factor
        self.window_len = window_len

        LRScheduler.__init__(self, optimizer, starting_step)

    def reset(self, starting_step: int = 0):
        LRScheduler.reset(self, starting_step)
        self.reset_window()
        self.update_args()

    def reset_window(self):
        self.window = [self.val_init] * self.window_len
        self.window_ptr = -1

    def step(self, value: float = None, increment: int = 1):
        self.step_ctr += increment

        if value is None:
            return

        self.window_ptr = (self.window_ptr + 1) % self.window_len
        self.window[self.window_ptr] = value

        val_avg = sum(self.window) / self.window_len

        if self.val_min <= val_avg <= self.val_max:
            return

        self.lrs = [max(self.lr_min, lr * self.down_scale) for lr in self.lrs]
        self.lr = sum(self.lrs) / len(self.lrs)

        self.reset_window()
        self.update_args()


class CoeffScheduler:
    step_ctr: int

    def __init__(
        self,
        step_total: int,
        val_milestones: 'tuple[float, float]',
        cosine: bool = True,
        starting_step: int = 0,
        device: 'str | torch.device' = 'cuda'
    ):
        self.step_total = step_total
        self.start_value, self.end_value = val_milestones
        self.cosine = cosine

        self.value = torch.tensor(self.start_value, dtype=torch.float32, device=device)
        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.update_value()

    def step(self, increment: int = 1):
        self.step_ctr += increment
        self.update_value()

    def update_value(self):
        ratio = min(1., self.step_ctr / max(1, self.step_total))

        if self.cosine:
            ratio = (1. - cos(pi * ratio)) / 2.

        with torch.no_grad():
            self.value.fill_(self.start_value + (self.end_value - self.start_value) * ratio)

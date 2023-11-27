"""NN optimisation"""

from math import cos, pi

import torch
from torch import Tensor
from torch.optim import Optimizer


class NAdamW(Optimizer):
    """Adam with Nesterov momentum and decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: 'tuple[float, float, float]' = (0.9, 0.9, 0.98),
        beta_products: 'tuple[float, float]' = (1., 1.),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        clip_grad_value: float = 4.,
        device: 'str | torch.device' = 'cuda'
    ):
        defaults = dict(
            lr=torch.tensor(lr, dtype=torch.float32, device=device),
            beta1=torch.tensor(betas[0], dtype=torch.float32, device=device),
            beta1_next=torch.tensor(betas[1], dtype=torch.float32, device=device),
            beta2=torch.tensor(betas[2], dtype=torch.float32, device=device),
            beta1_product=torch.tensor(beta_products[0], dtype=torch.float32, device=device),
            beta2_product=torch.tensor(beta_products[1], dtype=torch.float32, device=device),
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
                    grad.data.clip_(-clip_grad_value, clip_grad_value)

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
    step_ctr: int

    def __init__(self, optimiser: Optimizer, lr: float, starting_step: int = 0):
        self.optimiser = optimiser
        self.lr = lr
        self.reset(starting_step)

        with torch.no_grad():
            for param_group in self.optimiser.param_groups:
                param_group['lr'].fill_(lr)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step

    def step(self, _value: float = None, increment: int = 1):
        self.step_ctr += increment


class ConstantScheduler(LRScheduler):
    """Constant learning rate scheduler that only increments the step counter."""


class AnnealingScheduler(LRScheduler):
    """Linear or cosine decaying learning rate scheduler."""

    def __init__(
        self,
        optimiser: Optimizer,
        step_total: int,
        lr_milestones: 'tuple[float, float]' = (4e-4, 1e-6),
        beta1_milestones: 'tuple[float, float]' = (0.85, 0.98),
        cosine: bool = False,
        starting_step: int = 0
    ):
        self.optimiser = optimiser

        self.step_total = step_total
        self.lr_init, self.lr_final = lr_milestones
        self.beta1_init, self.beta1_final = beta1_milestones
        self.cosine = cosine

        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.update_params(*self.anneal())

    def step(self, _value: float = None, increment: int = 1):
        self.step_ctr += increment
        self.update_params(*self.anneal())

    def anneal(self) -> 'tuple[float, float, float]':
        ratio = self.get_ratio(self.step_ctr, self.step_total)
        ratio_next = self.get_ratio(self.step_ctr + 1, self.step_total)

        self.lr = self.lerp(self.lr_init, self.lr_final, ratio)
        beta1 = self.lerp(self.beta1_init, self.beta1_final, ratio)
        beta1_next = self.lerp(self.beta1_init, self.beta1_final, ratio_next)

        return self.lr, beta1, beta1_next

    def update_params(self, lr: float, beta1: float, beta1_next: float):
        with torch.no_grad():
            for param_group in self.optimiser.param_groups:
                param_group['lr'].fill_(lr)
                param_group['beta1'].fill_(beta1)
                param_group['beta1_next'].fill_(beta1_next)

    def get_ratio(self, num: int, den: int) -> float:
        ratio = max(0., min(1., num / max(1, den)))

        if self.cosine:
            ratio = (1. - cos(pi * ratio)) / 2.

        return ratio

    @staticmethod
    def lerp(start: float, end: float, ratio: float) -> float:
        return start + (end - start) * ratio


class OneCycleScheduler(AnnealingScheduler):
    """Two-phase annealing learning rate scheduler, with warmup and cooldown."""

    def __init__(
        self,
        optimiser: Optimizer,
        step_milestones: 'tuple[int, int]',
        lr_milestones: 'tuple[float, float, float]' = (2e-5, 4e-4, 1e-6),
        beta1_milestones: 'tuple[float, float, float]' = (0.9, 0.85, 0.98),
        cosine: bool = True,
        starting_step: int = 0
    ):
        self.optimiser = optimiser

        self.step_at_peak, self.step_total = step_milestones
        self.lr_init, self.lr_max, self.lr_final = lr_milestones
        self.beta1_init, self.beta1_min, self.beta1_final = beta1_milestones
        self.cosine = cosine

        self.reset(starting_step)

    def anneal(self) -> 'tuple[float, float, float]':
        if self.step_ctr < self.step_at_peak:
            ratio = self.get_ratio(self.step_ctr, self.step_at_peak)
            ratio_next = self.get_ratio(self.step_ctr + 1, self.step_at_peak)

            self.lr = self.lerp(self.lr_init, self.lr_max, ratio)
            beta1 = self.lerp(self.beta1_init, self.beta1_min, ratio)
            beta1_next = self.lerp(self.beta1_init, self.beta1_min, ratio_next)

        else:
            ratio = self.get_ratio(self.step_ctr, self.step_total)
            ratio_next = self.get_ratio(self.step_ctr + 1, self.step_total)

            self.lr = self.lerp(self.lr_max, self.lr_final, ratio)
            beta1 = self.lerp(self.beta1_min, self.beta1_final, ratio)
            beta1_next = self.lerp(self.beta1_min, self.beta1_final, ratio_next)

        return self.lr, beta1, beta1_next


class PlateauScheduler(AnnealingScheduler):
    """
    Tri-phase learning rate scheduler, separating warmup and cooldown with an
    extended middle elevation (plateau) to prolong learning at the maximum rate.
    """

    in_main: bool

    def __init__(
        self,
        optimiser: Optimizer,
        step_milestones: 'tuple[int, int, int]',
        lr_milestones: 'tuple[float, float, float]' = (2e-5, 4e-4, 1e-6),
        beta1_milestones: 'tuple[float, float, float]' = (0.9, 0.85, 0.98),
        cosine: bool = True,
        starting_step: int = 0
    ):
        self.optimiser = optimiser

        self.step_start_main, self.step_end_main, self.step_total = step_milestones
        self.lr_init, self.lr_max, self.lr_final = lr_milestones
        self.beta1_init, self.beta1_min, self.beta1_final = beta1_milestones
        self.cosine = cosine

        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.in_main = False
        self.update_params(*self.anneal())

    def step(self, _value: float = None, increment: int = 1):
        self.step_ctr += increment

        # Keep constant in main phase
        if self.in_main:
            if self.step_ctr < self.step_end_main:
                return

            self.in_main = False

        self.update_params(*self.anneal())

    def anneal(self) -> 'tuple[float, float, float]':
        if self.step_ctr < self.step_start_main:
            ratio = self.get_ratio(self.step_ctr, self.step_start_main)
            ratio_next = self.get_ratio(self.step_ctr + 1, self.step_start_main)

            self.lr = self.lerp(self.lr_init, self.lr_max, ratio)
            beta1 = self.lerp(self.beta1_init, self.beta1_min, ratio)
            beta1_next = self.lerp(self.beta1_init, self.beta1_min, ratio_next)

        elif self.step_ctr >= self.step_end_main:
            shifted_ctr = self.step_ctr - self.step_end_main
            shifted_total = self.step_total - self.step_end_main

            ratio = self.get_ratio(shifted_ctr, shifted_total)
            ratio_next = self.get_ratio(shifted_ctr + 1, shifted_total)

            self.lr = self.lerp(self.lr_max, self.lr_final, ratio)
            beta1 = self.lerp(self.beta1_min, self.beta1_final, ratio)
            beta1_next = self.lerp(self.beta1_min, self.beta1_final, ratio_next)

        else:
            self.in_main = True

            self.lr = self.lr_max
            beta1 = self.beta1_min
            beta1_next = self.beta1_min

        return self.lr, beta1, beta1_next


class AdaptiveScheduler(LRScheduler):
    """
    Increases or decreases the learning rate based on where the tracked value
    average falls between given extremes and in regard to the reference value.
    """

    window: 'list[float]'
    window_ptr: int

    def __init__(
        self,
        optimiser: Optimizer,
        lr_milestones: 'tuple[float, float, float]' = (1e-4, 5e-4, 1e-6),
        val_milestones: 'tuple[float, float, float]' = (0.1, 0.2, 0.05),
        scale_factors: 'tuple[float, float]' = (0.5, 1.2),
        window_len: int = 16,
        starting_step: int = 0
    ):
        self.optimiser = optimiser

        self.lr_init, self.lr_max, self.lr_min = lr_milestones
        self.val_target, self.val_max, self.val_min = val_milestones
        self.down_scale, self.up_scale = scale_factors
        self.window_len = window_len

        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.lr = self.lr_init
        self.update_params()
        self.reset_window()

    def reset_window(self):
        self.window = [self.val_target] * self.window_len
        self.window_ptr = -1

    def step(self, value: float = None, increment: int = 1):
        self.step_ctr += increment

        if value is None:
            return

        self.window_ptr = (self.window_ptr + 1) % self.window_len
        self.window[self.window_ptr] = value

        val_avg = sum(self.window) / self.window_len

        if self.val_min < val_avg < self.val_max:
            return

        if val_avg > self.val_max:
            self.lr = max(self.lr_min, self.lr * self.down_scale)

        else:
            self.lr = min(self.lr_max, self.lr * self.up_scale)

        self.reset_window()
        self.update_params()

    def update_params(self):
        with torch.no_grad():
            for param_group in self.optimiser.param_groups:
                param_group['lr'].fill_(self.lr)


class AdaptivePlateauScheduler(AnnealingScheduler):
    """
    Applies warmup, constance, and cooldown phases to the bounds of the
    adaptive learning rate scheduler.
    """

    def __init__(
        self,
        optimiser: Optimizer,
        step_milestones: 'tuple[int, int, int]',
        lr_milestones: 'tuple[float, float, float]' = (1e-4, 5e-4, 1e-6),
        beta1_milestones: 'tuple[float, float, float]' = (0.9, 0.85, 0.98),
        high_milestones: 'tuple[float, float, float]' = (0.05, 0.2, 0.01),
        low_milestones: 'tuple[float, float, float]' = (0.01, 0.05, 0.),
        scale_factors: 'tuple[float, float]' = (0.7, 1.2),
        cosine: bool = True,
        window_len: int = 16,
        starting_step: int = 0
    ):
        self.optimiser = optimiser

        self.step_start_main, self.step_end_main, self.step_total = step_milestones
        self.lr_init, self.lr_max, self.lr_min = lr_milestones
        self.beta1_init, self.beta1_min, self.beta1_final = beta1_milestones
        self.high_init, self.high_max, self.high_final = high_milestones
        self.low_init, self.low_max, self.low_final = low_milestones
        self.down_scale, self.up_scale = scale_factors
        self.cosine = cosine
        self.window_len = window_len

        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.in_main = False
        self.lr = self.lr_init
        *_, beta1, beta1_next = self.anneal()
        self.update_params(self.lr, beta1, beta1_next)
        self.reset_window()

    def reset_window(self):
        self.window = [self.val_target] * self.window_len
        self.window_ptr = -1

    def step(self, value: float = None, increment: int = 1):
        self.step_ctr += increment

        if value is not None:
            self.window_ptr = (self.window_ptr + 1) % self.window_len
            self.window[self.window_ptr] = value

        val_avg = sum(self.window) / self.window_len
        val_min, val_max, beta1, beta1_next = self.anneal()

        if val_avg > val_max:
            self.lr = max(self.lr_min, self.lr * self.down_scale)
            self.reset_window()

        elif val_avg < val_min:
            self.lr = min(self.lr_max, self.lr * self.up_scale)
            self.reset_window()

        elif self.in_main:
            if self.step_ctr < self.step_end_main:
                return

            self.in_main = False

        self.update_params(self.lr, beta1, beta1_next)

    def anneal(self) -> 'tuple[float, ...]':
        if self.step_ctr < self.step_start_main:
            ratio = self.get_ratio(self.step_ctr, self.step_start_main)
            ratio_next = self.get_ratio(self.step_ctr + 1, self.step_start_main)

            val_min = self.lerp(self.low_init, self.low_max, ratio)
            val_max = self.lerp(self.high_init, self.high_max, ratio)

            beta1 = self.lerp(self.beta1_init, self.beta1_min, ratio)
            beta1_next = self.lerp(self.beta1_init, self.beta1_min, ratio_next)

        elif self.step_ctr >= self.step_end_main:
            shifted_ctr = self.step_ctr - self.step_end_main
            shifted_total = self.step_total - self.step_end_main

            ratio = self.get_ratio(shifted_ctr, shifted_total)
            ratio_next = self.get_ratio(shifted_ctr + 1, shifted_total)

            val_min = self.lerp(self.low_max, self.low_final, ratio)
            val_max = self.lerp(self.high_max, self.high_final, ratio)

            beta1 = self.lerp(self.beta1_min, self.beta1_final, ratio)
            beta1_next = self.lerp(self.beta1_min, self.beta1_final, ratio_next)

        else:
            self.in_main = True

            val_min = self.low_max
            val_max = self.high_max

            beta1 = self.beta1_min
            beta1_next = self.beta1_min

        self.val_target = (val_min + val_max) / 2.

        return val_min, val_max, beta1, beta1_next

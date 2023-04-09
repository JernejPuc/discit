"""NN optimisation"""

from math import cos

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
            lr = group['lr']
            beta1 = group['beta1']
            beta1_next = group['beta1_next']
            beta2 = group['beta2']
            beta1_product = group['beta1_product']
            beta2_product = group['beta2_product']
            eps = group['eps']
            weight_mul = (1. - lr * group['weight_decay']) if group['weight_decay'] is not None else None
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

            grad_step = -lr * one_minus_beta1 / bias_correction1
            momentum_step = -lr * beta1_next / bias_correction1_next

            # Apply per param with valid grad
            for p in group['params']:
                if p.grad is None:
                    continue

                elif p.grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')

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


class SoftConstLRScheduler:
    """
    Adds cosine warmup and cooldown to a constant learning rate schedule,
    resembling a one-cycle scheduler with extended middle elevation (plateau)
    to prolong learning at the maximum rate.

    NOTE: Milestones refer to lr, beta1, and duration of the starting, main,
    and final phase.
    """

    step_ctr: int
    in_main: bool

    def __init__(
        self,
        optimiser: Optimizer,
        step_milestones: 'tuple[int, int, int]',
        starting_step: int = 0,
        lr_milestones: 'tuple[float, float, float]' = (2e-5, 4e-4, 1e-6),
        beta1_milestones: 'tuple[float, float, float]' = (0.9, 0.85, 0.98)
    ):
        self.optimiser = optimiser

        self.lr_init, self.lr_main, self.lr_final = lr_milestones
        self.beta1_init, self.beta1_main, self.beta1_final = beta1_milestones

        self.step_start_main = step_milestones[0]
        self.step_end_main = step_milestones[1] + self.step_start_main
        self.step_total = step_milestones[2] + self.step_end_main

        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.in_main = False

    def update_params(self, lr: float, beta1: float, beta1_next: float):
        with torch.no_grad():
            for param_group in self.optimiser.param_groups:
                param_group['lr'].fill_(lr)
                param_group['beta1'].fill_(beta1)
                param_group['beta1_next'].fill_(beta1_next)

    def anneal(self, start: float, end: float, phase_ratio: float) -> float:
        """Cosine anneal from start to end as phase_ratio goes from 0 to 1."""

        return end + (start - end) / 2. * (cos(torch.pi * phase_ratio) + 1.)

    def step(self, increment: int = 1):
        # Keep constant in main phase
        if self.in_main:
            if self.step_ctr < self.step_end_main:
                self.step_ctr += increment
                return

            else:
                self.in_main = False

        # Get annealed lr and momentum
        if self.step_ctr < self.step_start_main:
            phase_ratio = max(0., min(1., self.step_ctr / self.step_start_main))
            next_ratio = max(0., min(1., (self.step_ctr+1) / self.step_start_main))

            lr = self.anneal(self.lr_init, self.lr_main, phase_ratio)
            beta1 = self.anneal(self.beta1_init, self.beta1_main, phase_ratio)
            beta1_next = self.anneal(self.beta1_init, self.beta1_main, next_ratio)

        elif self.step_ctr >= self.step_end_main:
            phase_ratio = max(0., min(1., (self.step_ctr-self.step_end_main) / (self.step_total-self.step_end_main)))
            next_ratio = max(0., min(1., (self.step_ctr+1-self.step_end_main) / (self.step_total-self.step_end_main)))
            lr = self.anneal(self.lr_main, self.lr_final, phase_ratio)
            beta1 = self.anneal(self.beta1_main, self.beta1_final, phase_ratio)
            beta1_next = self.anneal(self.beta1_main, self.beta1_final, next_ratio)

        else:
            self.in_main = True
            lr = self.lr_main
            beta1 = self.beta1_main
            beta1_next = self.beta1_main

        # Update
        self.update_params(lr, beta1, beta1_next)
        self.step_ctr += increment

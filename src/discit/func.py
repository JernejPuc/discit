"""Functions"""

import torch
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    return x.sign() * (x.abs() + 1.).log()


def symexp(x: Tensor) -> Tensor:
    return x.sign() * (x.abs().exp() - 1.)


class TaLU(torch.autograd.Function):
    """Tangential linear unit with param. 1/10."""

    @staticmethod
    def forward(ctx, i: Tensor) -> Tensor:
        x = i.mul(10.).tanh_().div_(10.).clamp_min_(i)
        ctx.save_for_backward(i)

        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g: Tensor) -> Tensor:
        i, = ctx.saved_tensors
        x = i.mul(10.).clamp_max_(0.).cosh_().pow_(-2).mul_(g)

        return x


class GradDiscretise(torch.autograd.Function):
    """
    Output one hot samples instead of probabilities, but pass gradients
    through the latter.
    """

    @staticmethod
    def forward(probs: Tensor, one_hots: Tensor) -> Tensor:
        return one_hots

    @staticmethod
    def setup_context(ctx, inputs: 'tuple[Tensor, Tensor]', output: Tensor):
        ctx.save_for_backward(inputs[1])

    @staticmethod
    def backward(ctx, grad: Tensor) -> 'tuple[Tensor, None]':
        one_hots, = ctx.saved_tensors

        return grad * one_hots, None


class WeightedGradDiscretise(torch.autograd.Function):
    """
    Output one hot samples instead of probabilities, but pass gradients
    through the latter, weighted by importance.
    """

    @staticmethod
    def forward(probs: Tensor, one_hots: Tensor) -> Tensor:
        return one_hots

    @staticmethod
    def setup_context(ctx, inputs: 'tuple[Tensor, Tensor]', output: Tensor):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad: Tensor) -> 'tuple[Tensor, None]':
        probs, one_hots = ctx.saved_tensors

        return grad * (probs * one_hots), None

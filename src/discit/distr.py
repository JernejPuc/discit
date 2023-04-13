"""Distributions"""

from functools import cached_property
from math import log, pi, sqrt
from typing import Callable

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid, log_softmax


class Distribution:
    mean: Tensor
    mode: Tensor
    log_dev: Tensor
    dev: Tensor
    var: Tensor
    entropy: Tensor
    kl_div: 'Callable[[Distribution], Tensor]'
    log_prob: 'Callable[[Tensor, ...], Tensor]'
    prob: 'Callable[[Tensor, ...], Tensor]'
    sample: 'Callable[[], tuple[Tensor, ...]]'


class Continuous(Distribution):
    pass


class Discrete(Distribution):
    log_probs: Tensor
    probs: Tensor


class MultiMixed(Distribution):
    def __init__(self, mcat: 'MultiCategorical', mnor: 'MultiNormal'):
        self.mcat = mcat
        self.mnor = mnor

    @cached_property
    def mean(self) -> Tensor:
        return torch.cat((self.mcat.mean, self.mnor.mean), dim=-1)

    @cached_property
    def mode(self) -> Tensor:
        return torch.cat((self.mcat.mode, self.mnor.mode), dim=-1)

    @cached_property
    def log_dev(self) -> Tensor:
        return self.dev.log()

    @cached_property
    def dev(self) -> Tensor:
        return self.var.sqrt()

    @cached_property
    def var(self) -> Tensor:
        return torch.cat((self.mcat.var, self.mnor.var), dim=-1)

    @cached_property
    def entropy(self) -> Tensor:
        return self.mcat.entropy + self.mnor.entropy

    def kl_div(self, other: 'MultiMixed') -> Tensor:
        return self.mcat.kl_div(other.mcat) + self.mnor.kl_div(other.mnor)

    def log_prob(self, _values: Tensor, values_mnor: Tensor, indices_mcat: Tensor) -> Tensor:
        return self.mcat.log_prob(None, indices_mcat) + self.mnor.log_prob(values_mnor)

    def prob(self, _values: Tensor, values_mnor: Tensor, indices_mcat: Tensor) -> Tensor:
        return self.mcat.prob(None, indices_mcat) * self.mnor.prob(values_mnor)

    def sample(self) -> 'tuple[Tensor, Tensor, Tensor]':
        values_mcat, indices_mcat = self.mcat.sample()
        values_mnor, = self.mnor.sample()

        return torch.cat((values_mcat, values_mnor), dim=-1), values_mnor, indices_mcat


class MultiCategorical(Discrete):
    """
    Multiple independent categorical distributions multiplied and flattened
    to produce a single categorical distribution of larger dimension.
    """

    def __init__(self, values: Tensor, log_probs: Tensor, probs: Tensor = None):
        self.values = values
        self.log_probs = log_probs
        self.probs = log_probs.exp() if probs is None else probs

    @classmethod
    def from_raw(cls, *logituple: 'tuple[Tensor, ...]', values: Tensor) -> 'MultiCategorical':
        log_probs = log_softmax(logituple[0], dim=-1)

        if len(logituple) == 1:
            return cls(values, log_probs, logituple[0].softmax(dim=-1))

        for logits in logituple[1:]:
            log_probs = (log_probs.unsqueeze(-1) + log_softmax(logits, dim=-1).unsqueeze(-2)).flatten(-2)

        return cls(values, log_probs)

    @cached_property
    def mean(self) -> Tensor:
        return (self.values.unsqueeze(0) * self.probs.unsqueeze(-1)).sum(-2)

    @cached_property
    def mode(self) -> Tensor:
        indices = self.log_probs.argmax(dim=-1)

        return self.values[indices]

    @cached_property
    def log_dev(self) -> Tensor:
        return self.dev.log()

    @cached_property
    def dev(self) -> Tensor:
        return self.var.sqrt()

    @cached_property
    def var(self) -> Tensor:
        return ((self.values.unsqueeze(0)**2 * self.probs.unsqueeze(-1)).sum(-2) - self.mean**2).clip(0.)

    @cached_property
    def entropy(self) -> Tensor:
        return -(self.probs * self.log_probs).sum(-1)

    def kl_div(self, other: 'MultiCategorical') -> Tensor:
        return (self.probs * (self.log_probs - other.log_probs)).sum(-1)

    def log_prob(self, _values: Tensor, indices: Tensor) -> Tensor:
        return self.log_probs.gather(-1, indices).squeeze(-1)

    def prob(self, _values: Tensor, indices: Tensor) -> Tensor:
        return self.probs.gather(-1, indices).squeeze(-1)

    def sample(self) -> 'tuple[Tensor, Tensor]':
        indices = self.probs.multinomial(1)
        values = self.values.index_select(0, indices.squeeze(-1))

        return values, indices


class InterCategorical(MultiCategorical):
    """
    Categorical distribution where values stand for delimiters between bins.
    Probability of an arbitrary value is interpolated in proportion to
    probabilities of the two bounding values.
    """

    def log_prob(self, values: Tensor, _indices: Tensor = None) -> Tensor:
        values = values.squeeze(-1)
        delims = self.values.squeeze(-1)

        indices_above = torch.bucketize(values, delims).clip(1)
        indices_below = indices_above - 1

        values_above = delims.index_select(0, indices_above)
        values_below = delims.index_select(0, indices_below)

        ratio = (values - values_below) / (values_above - values_below)

        return torch.lerp(
            self.log_probs.gather(-1, indices_below.unsqueeze(-1)),
            self.log_probs.gather(-1, indices_above.unsqueeze(-1)),
            ratio.unsqueeze(-1)).squeeze(-1)

    def prob(self, values: Tensor, _indices: Tensor = None) -> Tensor:
        values = values.squeeze(-1)
        delims = self.values.squeeze(-1)

        indices_above = torch.bucketize(values, delims).clip(1)
        indices_below = indices_above - 1

        values_above = delims.index_select(0, indices_above)
        values_below = delims.index_select(0, indices_below)

        ratio = (values - values_below) / (values_above - values_below)

        return torch.lerp(
            self.probs.gather(-1, indices_below.unsqueeze(-1)),
            self.probs.gather(-1, indices_above.unsqueeze(-1)),
            ratio.unsqueeze(-1)).squeeze(-1)


class MultiNormal(Continuous):
    """
    Multivariate normal with diagonal covariance matrix,
    i.e. independent normal axes, where only diagonal elements are non-zero
    (no correlations between variables are intended).
    """

    _LOG_SQRT_2PI = 0.5 * (log(2.) + log(pi))
    _LOG_SQRT_2PIE = 0.5 + _LOG_SQRT_2PI
    _SQRT_2PI = sqrt(2. * pi)

    def __init__(self, mean: Tensor, log_dev: Tensor, dev: Tensor = None):
        self.mode = self.mean = mean
        self.log_dev = log_dev
        self.dev = log_dev.exp() if dev is None else dev

    @classmethod
    def from_raw(
        cls,
        mean: Tensor,
        pseudo_log_dev: Tensor,
        log_dev_bias: float = None,
        dev_bias: float = None,
        max_mean: float = None
    ) -> 'MultiNormal':

        if max_mean:
            mean = (mean / max_mean).tanh() * max_mean

        if log_dev_bias:
            pseudo_log_dev = pseudo_log_dev + log_dev_bias

        if dev_bias:
            dev = pseudo_log_dev.sigmoid() + dev_bias
            log_dev = dev.log()

        else:
            dev = pseudo_log_dev.sigmoid()
            log_dev = logsigmoid(pseudo_log_dev)

        return cls(mean, log_dev, dev)

    @cached_property
    def var(self) -> Tensor:
        return self.dev ** 2

    @cached_property
    def _double_var(self) -> Tensor:
        return 2. * self.var

    @cached_property
    def entropy(self) -> Tensor:
        return (self._LOG_SQRT_2PIE + self.log_dev).sum(-1)

    def kl_div(self, other: 'MultiNormal') -> Tensor:
        return (
            other.log_dev - self.log_dev - 0.5
            + (self.var + (self.mean - other.mean)**2) / other._double_var).sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        return -((values - self.mean)**2 / self._double_var + self._LOG_SQRT_2PI + self.log_dev).sum(-1)

    def prob(self, values: Tensor) -> Tensor:
        return ((-(values - self.mean)**2 / self._double_var).exp() / (self._SQRT_2PI * self.dev)).prod(-1)

    def sample(self) -> 'tuple[Tensor]':
        return torch.normal(self.mean, self.dev),


class ClippedNormal(MultiNormal):
    """
    Clipped independent multivariate normal distribution, where samples are
    bound to an interval and probabilities at the bounds integrate
    the overflowing density.
    """

    _LOG_05 = log(0.5)
    _SQRT_2 = sqrt(2.)

    def __init__(self, mean: Tensor, log_dev: Tensor, dev: Tensor = None, low: float = None, high: float = None):
        MultiNormal.__init__(self, mean, log_dev, dev)

        self.low = low
        self.high = high

    @classmethod
    def from_raw(
        cls,
        mean: Tensor,
        pseudo_log_dev: Tensor,
        log_dev_bias: float = None,
        dev_bias: float = None,
        max_mean: float = None,
        low: float = None,
        high: float = None
    ) -> 'ClippedNormal':

        if max_mean:
            mean = (mean / max_mean).tanh() * max_mean

        if log_dev_bias:
            pseudo_log_dev = pseudo_log_dev + log_dev_bias

        if dev_bias:
            dev = pseudo_log_dev.sigmoid() + dev_bias
            log_dev = dev.log()

        else:
            dev = pseudo_log_dev.sigmoid()
            log_dev = logsigmoid(pseudo_log_dev)

        return cls(mean, log_dev, dev, low, high)

    @cached_property
    def _sqrt_2_dev(self) -> Tensor:
        return self._SQRT_2 * self.dev

    def log_prob(self, values: Tensor) -> Tensor:
        log_prob = -((values - self.mean)**2 / self._double_var + self._LOG_SQRT_2PI + self.log_dev)

        if self.high:
            log_prob_high = self._LOG_05 + ((1. + 1e-6) - ((self.high - self.mean) / self._sqrt_2_dev).erf()).log()
            log_prob = log_prob.lerp(log_prob_high, (values > self.high).float())

        if self.low:
            log_prob_low = self._LOG_05 + ((1. + 1e-6) + ((self.low - self.mean) / self._sqrt_2_dev).erf()).log()
            log_prob = log_prob.lerp(log_prob_low, (values < self.low).float())

        return log_prob.sum(-1)

    def prob(self, values: Tensor) -> Tensor:
        prob = (-(values - self.mean)**2 / self._double_var).exp() / (self._SQRT_2PI * self.dev)

        if self.high:
            prob_high = 0.5 * ((1. + 1e-6) - ((self.high - self.mean) / self._sqrt_2_dev).erf())
            prob = prob.lerp(prob_high, (values > self.high).float())

        if self.low:
            prob_low = 0.5 * ((1. + 1e-6) + ((self.low - self.mean) / self._sqrt_2_dev).erf())
            prob = prob.lerp(prob_low, (values < self.low).float())

        return prob.prod(-1)

    def sample(self) -> 'tuple[Tensor]':
        return torch.normal(self.mean, self.dev).clamp_(self.low, self.high),


class FixedVarNormal(Continuous):
    """
    Normal distribution with fixed variance of 0.5, the log_prob of which
    corresponds to a slight shift of the negative mean squared error
    (with the same derivative).
    """

    _NEG_LOG_SQRT_PI = -0.5 * log(pi)
    _LOG_SQRT_2PIE = 0.5 * (log(2.) + log(pi) + 1.)
    _SQRT_PI = sqrt(pi)

    def __init__(self, mean: Tensor):
        self.mode = self.mean = mean

    @cached_property
    def log_dev(self) -> Tensor:
        return self.dev.log()

    @cached_property
    def dev(self) -> Tensor:
        return self.var.sqrt()

    @cached_property
    def var(self) -> Tensor:
        return torch.full_like(self.mean, 0.5)

    @cached_property
    def entropy(self) -> Tensor:
        return (self._LOG_SQRT_2PIE + self.log_dev).sum(-1)

    def kl_div(self, othr: 'FixedVarNormal') -> Tensor:
        return (self.mean - othr.mean).square().sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        return (-(self.mean - values)**2 + self._NEG_LOG_SQRT_PI).sum(-1)

    def prob(self, values: Tensor) -> Tensor:
        return ((-(self.mean - values)**2).exp() / self._SQRT_PI).prod(-1)

    def sample(self) -> 'tuple[Tensor]':
        return torch.normal(self.mean, self.dev),

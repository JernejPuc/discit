"""Distributions"""

import math
from functools import cached_property
from typing import Callable

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid, log_softmax, softplus


class ActionDistrTemplate:
    entropy: Tensor
    log_prob: 'Callable[[Tensor], Tensor]'
    kl_div: 'Callable[[ActionDistrTemplate], Tensor]'
    sample: 'Callable[[], Tensor]'


class ValueDistrTemplate:
    mean: Tensor
    log_prob: 'Callable[[Tensor], Tensor]'


class IndepNormal(ActionDistrTemplate):
    """
    Multivariate normal with diagonal covariance matrix,
    i.e. independent normal axes, where only diagonal elements are non-zero
    (no correlations between variables are intended).

    Includes arguments for bounding loc and scale to mitigate potential
    instabilities.
    """

    _LOG_SQRT_2PI = 0.5 * (math.log(2.) + math.log(math.pi))
    _LOG_SQRT_2PIE = 0.5 + _LOG_SQRT_2PI

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        sample: Tensor = None,
        pseudo: bool = False,
        tanh_arg: float = 3.,
        scale_min: float = 0.01,
        scale_bias: float = -math.log(3.)   # sigmoid(0 - log(3)) = 0.25
    ):
        self.scale_min = scale_min
        self.scale_bias = scale_bias

        # Shadow cached properties
        if not pseudo:
            self.mode = self.mean = self.loc = loc
            self.scale = scale
            self.pseudo_scale = None

        else:
            self.mode = self.mean = self.loc = (loc / tanh_arg).tanh() * tanh_arg
            self.pseudo_scale = scale

        if sample is not None:
            self.sample = sample

    @cached_property
    def log_scale(self) -> Tensor:
        return self.scale.log()

    @cached_property
    def scale(self) -> Tensor:
        return (self.pseudo_scale + self.scale_bias).sigmoid() + self.scale_min

    @cached_property
    def var(self) -> Tensor:
        return self.scale ** 2

    @cached_property
    def _dbl_var(self) -> Tensor:
        return 2. * self.var

    @cached_property
    def entropy(self) -> Tensor:
        return (self._LOG_SQRT_2PIE + self.log_scale).sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        return -(self._LOG_SQRT_2PI + self.log_scale + ((values - self.loc)**2) / self._dbl_var).sum(-1)

    def kl_div(self, othr: 'IndepNormal') -> Tensor:
        return (othr.log_scale - self.log_scale + ((self.var + (self.loc - othr.loc)**2) / othr._dbl_var) - 0.5).sum(-1)

    def sample(self) -> Tensor:
        return torch.normal(self.loc, self.scale)


class ClipIndepNormal(ActionDistrTemplate):
    """
    Clipped multivariate normal with diagonal covariance matrix,
    i.e. independent normal axes, where only diagonal elements are non-zero
    (no correlations between variables are intended).

    Includes arguments for bounding loc and scale to mitigate potential
    instabilities.

    TODO: Formulae.
    """

    _LOG_SQRT_2PI = 0.5 * (math.log(2.) + math.log(math.pi))
    _LOG_SQRT_2PIE = 0.5 + _LOG_SQRT_2PI
    _LOG_05 = math.log(0.5)
    _SQRT_2 = math.sqrt(2)

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        sample: Tensor = None,
        pseudo: bool = False,
        tanh_arg: float = 3.,
        scale_min: float = 0.01,
        scale_bias: float = -math.log(3.),  # sigmoid(0 - log(3)) = 0.25
        low: float = -1.,
        high: float = 1.
    ):
        self.scale_min = scale_min
        self.scale_bias = scale_bias
        self.low = low
        self.high = high

        # Shadow cached properties
        if not pseudo:
            self.mode = self.mean = self.loc = loc
            self.scale = scale
            self.pseudo_scale = None

        else:
            self.mode = self.mean = self.loc = (loc / tanh_arg).tanh() * tanh_arg
            self.pseudo_scale = scale

        if sample is not None:
            self.sample = sample

    @cached_property
    def log_scale(self) -> Tensor:
        return self.scale.log()

    @cached_property
    def scale(self) -> Tensor:
        return (self.pseudo_scale + self.scale_bias).sigmoid() + self.scale_min

    @cached_property
    def _sqrt_2_scale(self) -> Tensor:
        return self._SQRT_2 * self.scale

    @cached_property
    def var(self) -> Tensor:
        return self.scale ** 2

    @cached_property
    def _dbl_var(self) -> Tensor:
        return 2. * self.var

    @cached_property
    def entropy(self) -> Tensor:
        return (self._LOG_SQRT_2PIE + self.log_scale).sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        log_prob_b = self._LOG_05 + ((1. + 1e-6) - ((self.high - self.loc) / self._sqrt_2_scale).erf()).log()
        log_prob_a = self._LOG_05 + ((1. + 1e-6) + ((self.low - self.loc) / self._sqrt_2_scale).erf()).log()

        log_prob = -(self._LOG_SQRT_2PI + self.log_scale + ((values - self.loc)**2) / self._dbl_var)
        log_prob = log_prob.lerp(log_prob_b, (values > self.high).float())
        log_prob = log_prob.lerp(log_prob_a, (values < self.low).float())

        return log_prob.sum(-1)

    def kl_div(self, othr: 'ClipIndepNormal') -> Tensor:
        return (othr.log_scale - self.log_scale + ((self.var + (self.loc - othr.loc)**2) / othr._dbl_var) - 0.5).sum(-1)

    def sample(self) -> Tensor:
        return torch.normal(self.loc, self.scale).clamp_(self.low, self.high)


class TruncIndepNormal(ActionDistrTemplate):
    """
    Truncated multivariate normal with diagonal covariance matrix,
    i.e. independent normal axes, where only diagonal elements are non-zero
    (no correlations between variables are intended).

    Reference:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8947456/
    """

    _LOG_SQRT_2_DIV_PI = 0.5 * (math.log(2.) - math.log(math.pi))
    _LOG_SQRT_PIE_DIV_2 = 0.5 - _LOG_SQRT_2_DIV_PI
    _SQRT_2_DIV_PI = math.sqrt(2. / math.pi)
    _SQRT_2 = math.sqrt(2)

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        sample: Tensor = None,
        pseudo: bool = False,
        low: float = -1.,
        high: float = 1.
    ):
        self.loc = loc
        self.low = low
        self.high = high

        # Shadow cached properties
        if not pseudo:
            self.scale = scale
            self.pseudo_scale = None

        else:
            self.pseudo_scale = scale

        if sample is not None:
            self.sample = sample

    @cached_property
    def log_scale(self) -> Tensor:
        return self.scale.log() if self.pseudo_scale is None else logsigmoid(self.pseudo_scale)

    @cached_property
    def scale(self) -> Tensor:
        return torch.sigmoid(self.pseudo_scale)

    @cached_property
    def _sqr_scale(self) -> Tensor:
        return self.scale ** 2

    @cached_property
    def _dbl_sqr_scale(self) -> Tensor:
        return 2. * self._sqr_scale

    @cached_property
    def _arg_low(self) -> Tensor:
        return (self.low - self.loc) / self.scale

    @cached_property
    def _arg_high(self) -> Tensor:
        return (self.high - self.loc) / self.scale

    @cached_property
    def _sqr_args(self) -> 'tuple[Tensor, Tensor]':
        return (-0.5 * self._arg_low ** 2).exp(), (-0.5 * self._arg_high ** 2).exp()

    # NOTE: 1/2 * (1 + x) terms are left out
    @cached_property
    def _cdf_low(self) -> Tensor:
        return (self._arg_low / self._SQRT_2).erf()

    @cached_property
    def _cdf_high(self) -> Tensor:
        return (self._arg_high / self._SQRT_2).erf()

    @cached_property
    def _cdf_diff(self) -> Tensor:
        return self._cdf_high - self._cdf_low

    @cached_property
    def _log_cdf_diff(self) -> Tensor:
        return self._cdf_diff.log()

    @cached_property
    def _pdf_diff(self) -> Tensor:
        return (self._sqr_args[1] - self._sqr_args[0]) / self._cdf_diff * self._SQRT_2_DIV_PI

    @cached_property
    def _scaled_pdf_diff(self) -> Tensor:
        return (self.high * self._sqr_args[1] - self.low * self._sqr_args[0]) / self._cdf_diff * self._SQRT_2_DIV_PI

    @cached_property
    def mean(self) -> Tensor:
        return self.loc - self.scale * self._pdf_diff

    @cached_property
    def var(self) -> Tensor:
        return self._sqr_scale * (1. - self._scaled_pdf_diff - self._pdf_diff**2)

    @cached_property
    def entropy(self) -> Tensor:
        return (self._LOG_SQRT_PIE_DIV_2 + self.log_scale + self._log_cdf_diff - self._scaled_pdf_diff * 0.5).sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        log_prob = (
            self._LOG_SQRT_2_DIV_PI - self.log_scale
            - ((values - self.loc)**2) / self._dbl_sqr_scale
            - self._log_cdf_diff).sum(-1)

        return torch.where((self.low < values) & (values < self.high), log_prob, 0.)

    def kl_div(self, othr: 'TruncIndepNormal') -> Tensor:
        rel_loc_diff = othr.loc / othr._sqr_scale - self.loc / self._sqr_scale

        return (
            rel_loc_diff / 2.
            + othr._log_cdf_diff - self._log_cdf_diff
            - rel_loc_diff * self.mean
            - (1. / self._dbl_sqr_scale - 1. / othr._dbl_sqr_scale) * (self.var + self.mean**2)
        ).sum(-1)

    def sample(self) -> Tensor:
        # Sample in CDF space
        uni_sample = torch.rand_like(self.loc).mul_(self._cdf_diff).add_(self._cdf_low)

        # Quantile/inverse fn.
        sample = uni_sample.erfinv_().mul_(self._SQRT_2).mul_(self.scale).add_(self.loc)

        # Just in case
        # sample = sample.clamp_(low, high)

        return sample


class OnlyMean(ValueDistrTemplate):
    """Not a probability distribution, but is a simple and useful default."""

    def __init__(self, mean: Tensor):
        self.mean = mean

    # TODO: Term inconsistency
    def log_prob(self, values: Tensor) -> Tensor:
        return -(self.mean.flatten() - values)**2


class InterpCategorical(ValueDistrTemplate):
    """
    Categorical distribution, where values represent the delimiters between bins.
    Log probability of a value is interpolated in proportion to log probabilities
    of the two bounding values.
    """

    def __init__(self, pseudo_logits: Tensor, values: Tensor, values_as_indices: bool = False):
        self.pseudo_logits = pseudo_logits
        self.values = values
        self.values_as_indices = values_as_indices
        self.add_dims = tuple([None for _ in range(len(pseudo_logits.shape)-1)])

    @cached_property
    def _dim_match_values(self) -> Tensor:
        return self.values[self.add_dims] if self.add_dims else self.values

    @cached_property
    def logits(self) -> Tensor:
        return log_softmax(self.pseudo_logits, dim=-1)

    @cached_property
    def probs(self) -> Tensor:
        return torch.softmax(self.pseudo_logits, dim=-1)

    @cached_property
    def mean(self) -> Tensor:
        return (self._dim_match_values * self.probs).sum(-1)

    @cached_property
    def mode(self) -> Tensor:
        return self.values[..., self.pseudo_logits.argmax(dim=-1)]

    @cached_property
    def var(self) -> Tensor:
        return (self._dim_match_values**2 * self.probs).sum(-1) - self.mean**2

    @cached_property
    def entropy(self) -> Tensor:
        return -(self.probs * self.logits).sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        ridx = torch.bucketize(values, self.values)
        lidx = ridx-1

        if self.values_as_indices:
            rval = ridx
            lval = lidx

        else:
            rval = self.values[ridx]
            lval = self.values[lidx]

        ratio = (values - lval) / (rval - lval)

        return self.logits[..., lidx] * (1. - ratio) + self.logits[..., ridx] * ratio

    def kl_div(self, othr: 'InterpCategorical') -> Tensor:
        return (self.probs * (self.logits - othr.logits)).sum(-1)

    def sample(self) -> Tensor:
        return torch.multinomial(self.probs, 1)


class AsymmetricLaplace(ValueDistrTemplate):
    def __init__(self, loc: Tensor, pseudo_scale: Tensor, pseudo_skew: Tensor):
        self.mode = self.loc = loc
        self.scale = softplus(pseudo_scale)
        self.skew = softplus(pseudo_skew)

    @cached_property
    def lscale(self) -> Tensor:
        return self.scale * self.skew

    @cached_property
    def rscale(self) -> Tensor:
        return self.scale / self.skew

    @cached_property
    def _1_sub_sqr_skew(self) -> Tensor:
        return 1. - self.skew**2

    @cached_property
    def mean(self) -> Tensor:
        return self.loc + self._1_sub_sqr_skew * self.rscale

    @cached_property
    def var(self) -> Tensor:
        return (1. + self.skew**4) * self.rscale**2

    @cached_property
    def entropy(self) -> Tensor:
        return 1. + self._1_sub_sqr_skew.log() + self.rscale.log()

    def log_prob(self, value: Tensor) -> Tensor:
        d = value - self.loc
        d = -d.abs() / torch.where(d > 0., self.rscale, self.lscale)

        return d - (self.lscale + self.rscale).log()

    def kl_div(self, othr: 'AsymmetricLaplace') -> Tensor:
        # TODO:
        # https://stats.stackexchange.com/questions/436674/kl-divergence-between-two-asymmetric-laplace-distributions
        raise NotImplementedError

    def sample(self) -> Tensor:
        lexp, rexp = self.loc.new_empty(2, *self.loc.shape).exponential_()
        lexp = lexp.mul_(self.lscale)
        rexp = rexp.mul_(self.rscale)

        return rexp.sub_(lexp).add_(self.loc)

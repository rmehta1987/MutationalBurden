import torch
from pyro.distributions import TransformModule
import numbers
from torch.distributions import constraints
import math

class AffineTransform(TransformModule):
    r"""
    Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

    Args:
        loc (Tensor or float): Location parameter.
        scale (Tensor or float): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """
    bijective = True

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self._event_dim = event_dim

    @property
    def event_dim(self):
        return self._event_dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return AffineTransform(self.loc, self.scale, self.event_dim, cache_size=cache_size)

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False

        if isinstance(self.loc, numbers.Number) and isinstance(other.loc, numbers.Number):
            if self.loc != other.loc:
                return False
        else:
            if not (self.loc == other.loc).all().item():
                return False

        if isinstance(self.scale, numbers.Number) and isinstance(other.scale, numbers.Number):
            if self.scale != other.scale:
                return False
        else:
            if not (self.scale == other.scale).all().item():
                return False

        return True

    @property
    def sign(self):
        if isinstance(self.scale, numbers.Real):
            return 1 if float(self.scale) > 0 else -1 if float(self.scale) < 0 else 0
        return self.scale.sign()

    def _call(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        scale = self.scale
        if isinstance(scale, numbers.Real):
            result = torch.full_like(x, math.log(abs(scale)))
        else:
            result = torch.abs(scale).log()
        if self.event_dim:
            result_size = result.size()[:-self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[:-self.event_dim]
        return result.expand(shape)

    def forward_shape(self, shape):
        return torch.broadcast_shapes(shape,
                                      getattr(self.loc, "shape", ()),
                                      getattr(self.scale, "shape", ()))

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(shape,
                                      getattr(self.loc, "shape", ()),
                                      getattr(self.scale, "shape", ()))
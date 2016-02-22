"""The interface of bricks and some simple implementations."""
import logging

from theano import tensor
from blocks.bricks.base import Brick

logger = logging.getLogger(__name__)

from capy.theano_utils import normalize


# TODO: how to make this work for any shape??
class NormalizeActivations(Brick):
    r"""Normalize the inputs so each example has unit norm.
    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input plus optional bias

        """
        return normalize(input_)



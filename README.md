# The-Qiskit-Practical-Experiments.
An open-source SDK for working with quantum computers at the level of pulses, applications, circuits, and algorithms.

**The Qiskit Experiment 1:**

**Data Processing code**

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from qiskit_experiments.framework.store_init_args import StoreInitArgs

class DataAction(ABC, StoreInitArgs):
    """Abstract action done on measured data to process it.
    Each subclass of DataAction must define the way it formats, validates and processes data.
    """

    def __init__(self, validate: bool = True):
        """Create new node.
        Args:
            validate: If set to False the DataAction will not validate its input.
        """
    @abstractmethod
    def _process(self, data: np.ndarray) -> np.ndarray:
        """Applies the data processing step to the data.
        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.
                If the elements are ufloat objects consisting of a nominal value and
                a standard error.
        Returns:
            The processed data.
        """

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format and validate the input.
        Check that the given data has the correct structure. This method may
        additionally change the data type, e.g. converting a list to a numpy array.
        Args:
            
        Returns:
            The data that has been validated and formatted.
        """
        return data

    def __json_encode__(self) -> Dict[str, Any]:
        """Return the config dict for this node."""
        return dict(
            cls=type(self),
            args=tuple(getattr(self, "__init_args__", OrderedDict()).values()),
            kwargs=dict(getattr(self, "__init_kwargs__", OrderedDict())),
        )

    @classmethod
    def __json_decode__(cls, config: Dict[str, Any]) -> "DataAction":
        """Initialize a node from config dict."""
        init_args = config.get("args", tuple())
        init_kwargs = config.get("kwargs", dict())

        return cls(*init_args, **init_kwargs)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Call the data action of this node on the data.
        Args:
            data: A numpy array with arbitrary dtype. If the elements are ufloat objects
                consisting of a nominal value and a standard error, then the error propagation
                is done automatically.
        Returns:
            The processed data.
        """
        return self._process(self._format_data(data))

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate={self._validate})"


class TrainableDataAction(DataAction):
    """A base class for data actions that need training.
    .. note::
        The parameters of trainable nodes computed during training should be listed
        in the class method :meth:`._default_parameters`. These parameters
        are initialized at construction time and serialized together with the
        constructor arguments.
    """

    def __init__(self, validate: bool = True):
        """Create new node.
        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate=validate)
        self._parameters = self._default_parameters()

    @classmethod 
    def _default_parameters(cls) -> Options:
        """Parameters of trainable nodes.
        The parameters defined here should be assigned a `None` to
        indicate that the node has not been trained.
        """
        return Options()


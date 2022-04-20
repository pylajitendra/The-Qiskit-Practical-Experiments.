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




**The Qiskit Experiment 2:**

**The Data Base Service:**

**The Data Base Experiment Code:**


"""Stored data class."""

import warnings
import logging
import dataclasses
import uuid

from typing import Optional, List, Any, Union, Callable, Dict, Tuple
import copy
from concurrent import futures
from threading import Event
from functools import wraps
import traceback

from collections import deque
from datetime import datetime
import numpy as np

from matplotlib import pyplot
from qiskit import QiskitError
from qiskit.providers import Job, Backend, Provider
from qiskit.result import Result
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit_experiments.framework.json import ExperimentEncoder, ExperimentDecoder

from .database_service import DatabaseServiceV1
from .exceptions import DbExperimentDataError, DbExperimentEntryNotFound, DbExperimentEntryExists
from .db_analysis_result import DbAnalysisResultV1 as DbAnalysisResult
from .utils import (
    save_data,
    qiskit_version,
    plot_to_svg_bytes,
    ThreadSafeOrderedDict,
    ThreadSafeList,
)

LOG = logging.getLogger(__name__)


def do_auto_save(func: Callable):
    """Decorate the input function to auto save data."""

    @wraps(func)
    def _wrapped(self, *args, **kwargs):
        return_val = func(self, *args, **kwargs)
        if self.auto_save:
            self.save_metadata()
        return return_val

    return _wrapped


@contextlib.contextmanager
def service_exception_to_warning():
    """Convert an exception raised by experiment service to a warning."""
    try:
        yield
    except Exception:  # pylint: disable=broad-except
        LOG.warning("Experiment service operation failed: %s", traceback.format_exc())


class ExperimentStatus(enum.Enum):
    """Class for experiment status enumerated type."""

    EMPTY = "experiment data is empty"
    INITIALIZING = "experiment jobs are being initialized"
    RUNNING = "experiment jobs is actively running"
    CANCELLED = "experiment jobs or analysis has been cancelled"
    POST_PROCESSING = "experiment analysis is actively running"
    DONE = "experiment jobs and analysis have successfully run"
    ERROR = "experiment jobs or analysis incurred an error"

    def __json_encode__(self):
        return self.name

    @classmethod
    def __json_decode__(cls, value):
        return cls.__members__[value]  # pylint: disable=unsubscriptable-object


class AnalysisStatus(enum.Enum):
    """Class for analysis callback status enumerated type."""

    QUEUED = "analysis callback is queued"
    RUNNING = "analysis callback is actively running"
    CANCELLED = "analysis callback has been cancelled"
    DONE = "analysis callback has successfully run"
    ERROR = "analysis callback incurred an error"

    def __json_encode__(self):
        return self.name

    @classmethod
    def __json_decode__(cls, value):
        return cls.__members__[value]  # pylint: disable=unsubscriptable-object


@dataclasses.dataclass
class AnalysisCallback:
    """Dataclass for analysis callback status"""

    name: str = ""
    callback_id: str = ""
    status: AnalysisStatus = AnalysisStatus.QUEUED
    error_msg: Optional[str] = None
    event: Event = dataclasses.field(default_factory=Event)

    def __getstate__(self):
        # We need to remove the Event object from state when pickling
        # since events are not pickleable
        state = self.__dict__
        state["event"] = None
        return state

    def __json_encode__(self):
        return self.__getstate__()


class DbExperimentData:
    """Base common type for all versioned DbExperimentData classes.
    Note this class should not be inherited from directly, it is intended
    to be used for type checking.
    """

    version = 0


class DbExperimentDataV1(DbExperimentData):
    """Class to define and handle experiment data stored in a database.
    This class serves as a container for experiment related data to be stored
    in a database, which may include experiment metadata, analysis results,
    and figures.
    """

    version = 1
    verbose = True  # Whether to print messages to the standard output.
    _metadata_version = 1
    _job_executor = futures.ThreadPoolExecutor()

    _json_encoder = ExperimentEncoder
    _json_decoder = ExperimentDecoder

    def __init__(
        self,
        experiment_type: Optional[str] = "Unknown",
        backend: Optional[Backend] = None,
        
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        job_ids: Optional[List[str]] = None,
        share_level: Optional[str] = None,
        metadata: Optional[Dict] = None,
        
        **kwargs,
    ):
        """Initializes the DbExperimentData instance.
        Args:
            experiment_type: Experiment type.
            backend: Backend the experiment runs on.
            experiment_id: Experiment ID. One will be generated if not supplied.
            parent_id: The experiment ID of the parent experiment.
            This is applicable only if the database service supports sharing. See
            the specific service provider's documentation on valid values.
            metadata: Additional experiment metadata.
            figure_names: Name of figures associated with this experiment.
            notes: Freeform notes about the experiment.
            **kwargs: Additional experiment attributes.
        """
        metadata = metadata or {}
        self._metadata = copy.deepcopy(metadata)
        self._source = self._metadata.pop(
            "_source",
            {
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "metadata_version": self._metadata_version,
                "qiskit_version": qiskit_version(),
            },
        )

        self._service = service
        if self.service is None:
            self._set_service_from_backend(backend)
        self._backend = backend
        self._auto_save = False

        self._id = experiment_id or str(uuid.uuid4())
        self._parent_id = parent_id
        self._type = experiment_type
        self._tags = tags or []
        self._share_level = share_level
        self._notes = notes or ""

        self._jobs = ThreadSafeOrderedDict(job_ids or [])
        self._job_futures = ThreadSafeOrderedDict()
        self._analysis_callbacks = ThreadSafeOrderedDict()
        self._analysis_futures = ThreadSafeOrderedDict()
        # Set 2 workers for analysis executor so there can be 1 actively running
        # future and one waiting "running" future. This is to allow the second
        # future to be cancelled without waiting for the actively running future
        # to finish first.
        self._analysis_executor = futures.ThreadPoolExecutor(max_workers=2)
        self._monitor_executor = futures.ThreadPoolExecutor()

       
        self._created_in_db = False
        self._extra_data = kwargs




**The Qiskit Practical Experiment 3:**

**The Randomized Bench Marking:**

**The Randomized Bench Marking Experiment Code:**

"""
Standard RB Experiment class.
"""
from typing import Union, Iterable, Optional, List, Sequence

import numpy as np
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QiskitError
from qiskit.quantum_info import Clifford


from qiskit_experiments.framework import BaseExperiment, ParallelExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from .rb_analysis import RBAnalysis

from .rb_utils import RBUtils


class StandardRB(BaseExperiment, RestlessMixin):
    """Standard randomized benchmarking experiment.
    # section: overview
        Randomized Benchmarking (RB) is an efficient and robust method
        for estimating the average error-rate of a set of quantum gate operations.
        
        for an explanation on the RB method.
        A standard RB experiment generates sequences of random Cliffords
        such that the unitary computed by the sequences is the identity.
        After running the sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits an exponentially decaying curve, and estimates
        the Error Per Clifford (EPC), as described in Refs. [1, 2].
        See :class:`RBUtils` documentation for additional information
        on estimating the Error Per Gate (EPG) for 1-qubit and 2-qubit gates,
        from 1-qubit and 2-qubit standard RB experiments, by Ref. [3].
    # section: analysis_ref
        :py:class:`RBAnalysis`
    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887
        
    """

    def __init__(
        self,
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: Optional[bool] = False,
    ):
        """Initialize a standard randomized benchmarking experiment.
        Args:
            qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
       
        # Initialize base experiment
        super().__init__(qubits, analysis=RBAnalysis(), backend=backend)
        self._verify_parameters(lengths, num_samples)

        # Set configurable options
        self.set_experiment_options(lengths=list(lengths), num_samples=num_samples, seed=seed)
        self.analysis.set_options(outcome="0" * self.num_qubits)

        # Set fixed options
        self._full_sampling = full_sampling
        self._clifford_utils = CliffordUtils()

    def _verify_parameters(self, lengths, num_samples):
        """Verify input correctness, raise QiskitError if needed"""
        if any(length <= 0 for length in lengths):
            raise QiskitError(
                f"The lengths list {lengths} should only contain " "positive elements."
            )
        if len(set(lengths)) != len(lengths):
            raise QiskitError(
                f"The lengths list {lengths} should not contain " "duplicate elements."
            )
        if num_samples <= 0:
            raise QiskitError(f"The number of samples {num_samples} should " "be positive.")

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.
        Experiment Options:
            lengths (List[int]): A list of RB sequences lengths.
            num_samples (int): Number of samples to generate for each sequence length.
            seed (None or int or SeedSequence or BitGenerator or Generator): A seed
                used to initialize ``numpy.random.default_rng`` when generating circuits.
                The ``default_rng`` will be initialized with this seed value everytime
                :meth:`circuits` is called.
        """
        options = super()._default_experiment_options()

        options.lengths = None
        options.num_samples = None
        options.seed = None

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.
        Returns:
            A list of :class:`QuantumCircuit`.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []
        for _ in range(self.experiment_options.num_samples):
            circuits += self._sample_circuits(self.experiment_options.lengths, rng)
        return circuits

    def _sample_circuits(self, lengths: Iterable[int], rng: Generator) -> List[QuantumCircuit]:
        """Return a list RB circuits for the given lengths.
        Args:
            lengths: A list of RB sequences lengths.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
        Returns:
            A list of :class:`QuantumCircuit`.
        """
        circuits = []
        for length in lengths if self._full_sampling else [lengths[-1]]:
            elements = self._clifford_utils.random_clifford_circuits(self.num_qubits, length, rng)
            element_lengths = [len(elements)] if self._full_sampling else lengths
            circuits += self._generate_circuit(elements, element_lengths)
        return circuits

    def _generate_circuit(
        self, elements: Iterable[Clifford], lengths: Iterable[int]
    ) -> List[QuantumCircuit]:
        """Return the RB circuits constructed from the given element list.
        Args:
            elements: A list of Clifford elements
            lengths: A list of RB sequences lengths.
        Returns:
            A list of :class:`QuantumCircuit`s.
        Additional information:
            The circuits are constructed iteratively; each circuit is obtained
            by extending the previous circuit (without the inversion and measurement gates)
        """
        qubits = list(range(self.num_qubits))
        circuits = []

        circs = [QuantumCircuit(self.num_qubits) for _ in range(len(lengths))]
        for circ in circs:
            circ.barrier(qubits)
        circ_op = Clifford(np.eye(2 * self.num_qubits))

        for current_length, group_elt_circ in enumerate(elements):
            if isinstance(group_elt_circ, tuple):
                group_elt_gate = group_elt_circ[0]
                group_elt_op = group_elt_circ[1]
            else:
                group_elt_gate = group_elt_circ
                group_elt_op = Clifford(group_elt_circ)

            if not isinstance(group_elt_gate, Gate):
                group_elt_gate = group_elt_gate.to_gate()
            circ_op = circ_op.compose(group_elt_op)
            for circ in circs:
                circ.append(group_elt_gate, qubits)
                circ.barrier(qubits)
            if current_length + 1 in lengths:
                # copy circuit and add inverse
                inv = circ_op.adjoint()
                }
                rb_circ.measure_all()
                circuits.append(rb_circ)
        return circuits

    def _get_circuit_metadata(self, circuit):
        if circuit.metadata["experiment_type"] == self._type:
            return circuit.metadata
        if circuit.metadata["experiment_type"] == ParallelExperiment.__name__:
            for meta in circuit.metadata["composite_metadata"]:
                if meta["physical_qubits"] == self.physical_qubits:
                    return meta
        return None

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        transpiled = super()._transpiled_circuits()
        for c in transpiled:
            meta = self._get_circuit_metadata(c)
        return transpiled

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata



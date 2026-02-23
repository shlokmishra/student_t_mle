# Location models: Student, Laplace, Logistic.
# Each module provides get_mle, sample_data, get_benchmark_mle_samples, run_gibbs.

from . import student
from . import laplace
from . import logistic

__all__ = ["student", "laplace", "logistic"]

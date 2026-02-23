# Location models: Student, Laplace, Logistic.
# Each module provides get_mle, sample_data, get_benchmark_mle_samples, run_gibbs.

from . import loc_student
from . import loc_laplace
from . import loc_logistic

__all__ = ["loc_student", "loc_laplace", "loc_logistic"]

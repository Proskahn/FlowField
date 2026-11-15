"""Sequential TuRBO-1 with ARD local GPs and joint Thompson sampling.

Algorithm: Eriksson et al., Scalable Global Optimization via Local Bayesian
Optimization (NeurIPS 2019). This CPU implementation uses scikit-learn, one
region, and one evaluation per ask; it is not the multi-region TuRBO-m method.
"""
from dataclasses import dataclass
import math
from numbers import Real
import warnings

from .bo import dependencies
from .samplers import Sampler


def turbo_settings(dimension, initial_points=5, candidates=512):
    if type(initial_points) is not int or initial_points < 2:
        raise ValueError('turbo_initial_points must be an integer of at least 2 (including baseline)')
    if type(candidates) is not int or candidates < 16:
        raise ValueError('turbo_candidates must be an integer of at least 16')
    return dict(initial_points=initial_points, candidates=candidates, batch_size=1,
                initialization='baseline plus Latin hypercube; fresh Latin hypercube on restart',
                kernel='constant times ARD Matern-5/2', gp_restarts=2, jitter=1e-8,
                input_scaling='unit box', output_scaling='standardized local scalar objective',
                acquisition='joint posterior Thompson sampling, minimization',
                candidates_method='scrambled Sobol with sparse coordinate perturbations',
                perturb_probability=min(20./dimension, 1.),
                length_initial=.8, length_min=2.**-7, length_max=1.6,
                success_tolerance=3, failure_tolerance=max(4, dimension), improvement_threshold=1e-3,
                restart='discard local GP data; retain global journal and best result',
                constraint_handling='problem scalar penalties; explicit CFD failure penalties included')


@dataclass
class TrustRegionState:
    dimension: int
    length: float = .8
    successes: int = 0
    failures: int = 0
    best: float | None = None
    restart_required: bool = False

    def observe(self, value, *, adapt):
        if adapt and self.best is not None:
            if value < self.best - 1e-3*abs(self.best):
                self.successes += 1
                self.failures = 0
            else:
                self.failures += 1
                self.successes = 0
            if self.successes == 3:
                self.length = min(1.6, 2*self.length)
                self.successes = 0
            elif self.failures == max(4, self.dimension):
                self.length *= .5
                self.failures = 0
            self.restart_required = self.length < 2.**-7
        self.best = value if self.best is None else min(self.best, value)


def posterior_sample(model, candidates, rng):
    """One joint GP draw: correlations between candidate values are essential."""
    import numpy as np
    from scipy.linalg import cholesky
    mean, covariance = model.predict(candidates, return_cov=True)
    covariance = (covariance + covariance.T)*.5
    diagonal = np.diag_indices_from(covariance)
    scale = max(float(np.max(np.diag(covariance))), 1.)
    for jitter in (1e-10, 1e-8, 1e-6):
        stabilized = covariance.copy()
        stabilized[diagonal] += jitter*scale
        try:
            factor = cholesky(stabilized, lower=True)
            return mean + factor @ rng.standard_normal(len(candidates))
        except np.linalg.LinAlgError:
            continue
    raise RuntimeError('TuRBO posterior covariance could not be factorized')


class TuRBOSampler(Sampler):
    """Seeded TuRBO-1; population and sigma are unused common factory options."""
    def __init__(self, problem, *, seed, initial, population=4, sigma=.1,
                 turbo_initial_points=5, turbo_candidates=512):
        self.settings = turbo_settings(problem.dimension, turbo_initial_points, turbo_candidates)
        (self.np, _, self.ConvergenceWarning, self.GP,
         self.ConstantKernel, self.Matern) = dependencies()
        from scipy.stats import qmc
        self.Sobol = qmc.Sobol
        if problem.dimension > self.Sobol.MAXDIM:
            raise ValueError(f'TuRBO Sobol candidates support at most {self.Sobol.MAXDIM} variables')
        np = self.np
        self.problem = problem
        self.initial = list(problem.validate_design(initial))
        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.span = np.asarray(problem.upper_bounds, dtype=float) - self.lower
        if not np.all(np.isfinite(self.span)) or np.any(self.span <= 0):
            raise ValueError('TuRBO requires finite, nonzero real-variable bounds')
        self.rng = np.random.RandomState(seed)
        self.state = TrustRegionState(problem.dimension)
        self.restarts = 0
        self.x, self.y, self.seen = [], [], []
        self.initialized = False
        self.pending = None
        self.model = None
        self.warmup = self._latin(turbo_initial_points - 1)
        self.warmup_index = 0
        self._diagnostics()

    def _latin(self, count):
        return (self.np.column_stack([self.rng.permutation(count) for _ in range(self.problem.dimension)])
                + self.rng.random_sample((count, self.problem.dimension))) / count

    def _diagnostics(self):
        self.diagnostics = dict(turbo_state=dict(
            restarts=self.restarts, length=self.state.length, successes=self.state.successes,
            failures=self.state.failures, restart_required=self.state.restart_required,
            local_evaluations=len(self.y), local_best=self.state.best))

    def _observe(self, x, result, *, adapt):
        value = result['objective']
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError('TuRBO requires a finite scalar objective or explicit finite failure penalty')
        point = ((self.np.asarray(self.problem.validate_design(x)) - self.lower) / self.span).tolist()
        self.x.append(point)
        self.y.append(float(value))
        self.seen.append(point)
        self.state.observe(float(value), adapt=adapt)
        self._diagnostics()

    def initialize(self, x, result):
        if self.initialized or list(x) != self.initial:
            raise ValueError('TuRBO must be initialized once with its baseline design')
        self._observe(x, result, adapt=False)
        self.initialized = True

    def _novel(self, candidates):
        mask = self.np.ones(len(candidates), dtype=bool)
        for point in self.seen:
            mask &= self.np.max(self.np.abs(candidates - point), axis=1) > 1e-9
        return mask

    def _restart(self):
        self.restarts += 1
        self.state = TrustRegionState(self.problem.dimension)
        self.x, self.y = [], []
        self.model = None
        self.warmup = self._latin(self.settings['initial_points'])
        self.warmup_index = 0
        self._diagnostics()

    def _fit(self):
        np = self.np
        y = np.asarray(self.y)
        scaled = y / max(float(np.max(np.abs(y))), 1.)
        spread = float(np.std(scaled))
        targets = (scaled - np.mean(scaled)) / (spread if spread > 0 else 1.)
        kernel = self.ConstantKernel(1., (1e-3, 1e3)) * self.Matern(
            np.full(self.problem.dimension, .5), (.005, 4.), nu=2.5)
        self.model = self.GP(kernel=kernel, alpha=self.settings['jitter'], normalize_y=False,
                             n_restarts_optimizer=self.settings['gp_restarts'], random_state=self.rng)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', self.ConvergenceWarning)
            self.model.fit(np.asarray(self.x), targets)

    def _candidates(self):
        np = self.np
        dimension, count = self.problem.dimension, self.settings['candidates']
        center = np.asarray(self.x[int(np.argmin(self.y))])
        lengths = np.asarray(self.model.kernel_.k2.length_scale)
        weights = lengths / np.exp(np.mean(np.log(lengths)))
        lower = np.clip(center - .5*self.state.length*weights, 0., 1.)
        upper = np.clip(center + .5*self.state.length*weights, 0., 1.)
        engine = self.Sobol(d=dimension, scramble=True, seed=int(self.rng.randint(2**31)))
        perturbations = lower + (upper-lower)*engine.random_base2((count-1).bit_length())[:count]
        mask = self.rng.random_sample((count, dimension)) < self.settings['perturb_probability']
        empty = np.flatnonzero(~mask.any(axis=1))
        mask[empty, self.rng.randint(dimension, size=len(empty))] = True
        pool = np.where(mask, perturbations, center)
        return pool[self._novel(pool)]

    def ask(self):
        if not self.initialized:
            raise RuntimeError('Initialize TuRBO with the baseline before ask')
        if self.pending is not None:
            raise RuntimeError('tell must complete the previous TuRBO evaluation before ask')
        if self.state.restart_required:
            self._restart()
        self.pending_adapt = self.warmup_index >= len(self.warmup)
        if not self.pending_adapt:
            point = self.warmup[self.warmup_index]
            for _ in range(1000):
                if self._novel(self.np.asarray([point]))[0]:
                    break
                point = self.rng.random_sample(self.problem.dimension)
            else:
                self.stop_reason = 'no_new_design_found'
                return []
        else:
            self._fit()
            pool = self._candidates()
            if not len(pool):
                self.stop_reason = 'no_new_design_found'
                return []
            point = pool[int(self.np.argmin(posterior_sample(self.model, pool, self.rng)))]
        design = self.np.clip(self.lower + self.span*point, self.lower, self.lower+self.span).tolist()
        self.pending = [design]
        return [list(design)]

    def tell(self, designs, results):
        if self.pending is None or designs != self.pending or len(results) != 1:
            raise ValueError('TuRBO feedback must match the pending design')
        self._observe(designs[0], results[0], adapt=self.pending_adapt)
        if not self.pending_adapt:
            self.warmup_index += 1
        self.pending = None

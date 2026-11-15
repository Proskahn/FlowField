"""Sequential vanilla Bayesian optimization of the problem's scalar objective.

The surrogate is an exact GP with a fitted isotropic Matérn-5/2 kernel.
Optional constraint repair is supplied by the problem before acquisition scoring.
"""
import math
from numbers import Real
import warnings

from .samplers import Sampler
from .repair import CandidateRepair


def bo_settings(initial_points=5, candidates=2048):
    if type(initial_points) is not int or initial_points < 2:
        raise ValueError('bo_initial_points must be an integer of at least 2 (including baseline)')
    if type(candidates) is not int or candidates < 16:
        raise ValueError('bo_candidates must be an integer of at least 16')
    return dict(initial_points=initial_points, candidates=candidates,
                initialization='baseline plus Latin hypercube',
                kernel='constant times isotropic Matern-5/2',
                input_scaling='unit box', output_scaling='standardized scalar objective',
                acquisition='expected improvement, minimization', xi=.01,
                gp_restarts=2, jitter=1e-8,
                acquisition_restarts=5, local_search_max_dimension=20,
                constraint_handling='problem scalar penalties; explicit CFD failure penalties included')


def dependencies():
    try:
        import numpy as np
        from scipy.optimize import minimize
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    except ImportError as exc:
        raise RuntimeError('Install Bayesian optimization with: python -m pip install -e ".[bo]"') from exc
    return np, minimize, ConvergenceWarning, GaussianProcessRegressor, ConstantKernel, Matern


def expected_improvement(mean, std, best, xi=.01):
    """EI for minimization, including the deterministic zero-variance limit."""
    import numpy as np
    from scipy.special import ndtr
    mean, std = np.broadcast_arrays(np.asarray(mean, dtype=float), np.asarray(std, dtype=float))
    improvement = best - mean - xi
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
    ei = improvement * ndtr(z) + std * np.exp(-.5*z*z) / math.sqrt(2*math.pi)
    return np.maximum(np.where(std > 0, ei, improvement), 0.)


class BayesianOptimizationSampler(Sampler):
    """One proposal at a time; all randomness is private and replayable.

    Population and sigma are accepted for the common factory interface and are
    unused. Finite failure penalties are observations of the configured scalar
    objective, just as for CMA-ES; this is not constrained Bayesian optimization.
    """
    def __init__(self, problem, *, seed, initial, population=4, sigma=.1,
                 bo_initial_points=5, bo_candidates=2048):
        self.settings = bo_settings(bo_initial_points, bo_candidates)
        (self.np, self.minimize, self.ConvergenceWarning, self.GP,
         self.ConstantKernel, self.Matern) = dependencies()
        np = self.np
        self.problem = problem
        self.initial = list(problem.validate_design(initial))
        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.span = np.asarray(problem.upper_bounds, dtype=float) - self.lower
        if not np.all(np.isfinite(self.span)) or np.any(self.span <= 0):
            raise ValueError('BO requires finite, nonzero real-variable bounds')
        self.repair = CandidateRepair(problem)
        self.repair.check_initial(self.initial)
        self.rng = np.random.RandomState(seed)
        count, dimension = bo_initial_points - 1, problem.dimension
        self.warmup = (np.column_stack([self.rng.permutation(count) for _ in range(dimension)])
                       + self.rng.random_sample((count, dimension))) / count
        self.x, self.y = [], []
        self.pending = None
        self.model = None
        self.proposal_metadata = None

    def _observe(self, x, result):
        value = result['objective']
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError('BO requires a finite scalar objective or explicit finite failure penalty')
        normalized = (self.np.asarray(self.problem.validate_design(x)) - self.lower) / self.span
        self.x.append(normalized.tolist())
        self.y.append(float(value))

    def initialize(self, x, result):
        if self.x or list(x) != self.initial:
            raise ValueError('BO must be initialized once with its baseline design')
        self._observe(x, result)

    def _novel(self, candidates):
        # Avoid an n_candidates x n_observations x dimension temporary array.
        np = self.np
        mask = np.ones(len(candidates), dtype=bool)
        for previous in self.x:
            mask &= np.max(np.abs(candidates - previous), axis=1) > 1e-9
        return mask

    def _fit(self):
        np = self.np
        y = np.asarray(self.y)
        # Scaling first avoids overflow for large, explicitly chosen penalties.
        magnitude = max(float(np.max(np.abs(y))), 1.)
        scaled = y / magnitude
        spread = float(np.std(scaled))
        self.targets = (scaled - np.mean(scaled)) / (spread if spread > 0 else 1.)
        length = .2 * math.sqrt(self.problem.dimension)
        kernel = self.ConstantKernel(1., (1e-3, 1e3)) * self.Matern(
            length, (1e-3, 1e3 * math.sqrt(self.problem.dimension)), nu=2.5)
        self.model = self.GP(kernel=kernel, alpha=self.settings['jitter'],
                             normalize_y=False, n_restarts_optimizer=self.settings['gp_restarts'],
                             random_state=self.rng)
        with warnings.catch_warnings():
            # Bound optima are normal; do not flood the CLI with kernel tuning warnings.
            warnings.simplefilter('ignore', self.ConvergenceWarning)
            self.model.fit(np.asarray(self.x), self.targets)

    def _acquisition(self, candidates):
        mean, std = self.model.predict(candidates, return_std=True)
        return expected_improvement(mean, std, min(self.targets), self.settings['xi']), std

    def ask(self):
        if not self.x:
            raise RuntimeError('Initialize BO with the baseline before ask')
        if self.pending is not None:
            raise RuntimeError('tell must complete the previous BO evaluation before ask')
        np = self.np
        if len(self.x) < self.settings['initial_points']:
            proposed = self.warmup[len(self.x) - 1]
            for _ in range(1000):
                point = self.repair.apply(np.asarray([proposed]))[0]
                if self._novel(np.asarray([point]))[0]:
                    break
                proposed = self.rng.random_sample(self.problem.dimension)
            else:
                self.stop_reason = 'no_new_design_found'
                return []
        else:
            self._fit()
            originals = self.rng.random_sample((self.settings['candidates'], self.problem.dimension))
            pool = self.repair.apply(originals)
            novel = self._novel(pool)
            pool, originals = pool[novel], originals[novel]
            unique = self.repair.distinct(pool)
            pool, originals = pool[unique], originals[unique]
            if not len(pool):
                self.stop_reason = 'no_new_design_found'
                return []
            ei, std = self._acquisition(pool)
            order = np.argsort(-ei, kind='stable')
            refined, refined_originals = [], []
            if self.problem.dimension <= self.settings['local_search_max_dimension']:
                for start in pool[order[:self.settings['acquisition_restarts']]]:
                    fit = self.minimize(lambda x: -float(self._acquisition(
                        self.repair.apply(np.asarray([x])))[0][0]),
                                        start, method='L-BFGS-B', bounds=[(0., 1.)]*self.problem.dimension,
                                        options={'maxiter': 75})
                    proposed = np.clip(fit.x, 0., 1.)
                    if not np.all(np.isfinite(proposed)):
                        continue
                    point = self.repair.apply(np.asarray([proposed]))[0]
                    if self._novel(np.asarray([point]))[0]:
                        refined.append(point)
                        refined_originals.append(proposed)
            if refined:
                pool = np.vstack([pool, refined])
                originals = np.vstack([originals, refined_originals])
                unique = self.repair.distinct(pool)
                pool, originals = pool[unique], originals[unique]
                ei, std = self._acquisition(pool)
            # If EI underflows everywhere, explore posterior uncertainty.
            index = int(np.argmax(ei if np.max(ei) > 0 else std))
            point, proposed = pool[index], originals[index]
        design = np.clip(self.lower + self.span*point, self.lower, self.lower + self.span).tolist()
        self.pending = [design]
        self.proposal_metadata = self.repair.metadata(proposed, point)
        return [list(design)]

    def tell(self, designs, results):
        if self.pending is None or designs != self.pending or len(results) != 1:
            raise ValueError('BO feedback must match the pending design')
        self._observe(designs[0], results[0])
        self.pending = None

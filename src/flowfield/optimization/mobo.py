"""Sequential BoTorch qLogNEHVI and qLogNParEGO on named raw objectives.

All problem objectives are minimized; BoTorch receives their negatives.
Constraints remain g(x) <= 0. Acquisition functions are maximized over a finite
pool of repaired candidates, so non-differentiable geometry repair is supported.
The acquisition reference point is not a benchmark hypervolume scoring policy.
"""

import math
from numbers import Real

from .nsga2 import ObjectiveSpace
from .repair import CandidateRepair
from .samplers import Sampler


MOBO_METHODS = ('qlognehvi', 'qlognparego')


def mobo_settings(method, initial_points=5, candidates=2048, mc_samples=128,
                  reference_point=None):
    if method not in MOBO_METHODS:
        raise ValueError('Unknown BoTorch multi-objective method')
    if type(initial_points) is not int or initial_points < 2:
        raise ValueError('bo_initial_points must be an integer >= 2, including baseline')
    if type(candidates) is not int or candidates < 16:
        raise ValueError('bo_candidates must be an integer >= 16')
    if type(mc_samples) is not int or mc_samples < 16 or mc_samples & (mc_samples-1):
        raise ValueError('mobo_mc_samples must be a power of two >= 16')
    if reference_point is not None:
        if method != 'qlognehvi':
            raise ValueError('mobo_reference_point is only used by qlognehvi')
        if (not isinstance(reference_point, dict) or len(reference_point) < 2
                or any(not isinstance(k, str) or not k for k in reference_point)
                or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                       for v in reference_point.values())):
            raise ValueError('mobo_reference_point must map at least two objective names to finite values')
        reference_point = {k: float(v) for k, v in sorted(reference_point.items())}
    return dict(initial_points=initial_points, candidates=candidates, mc_samples=mc_samples,
        batch_size=1, device='cpu', dtype='float64', torch_threads=1,
        initialization='baseline plus Latin hypercube; additional uniform samples until two valid observations',
        model='BoTorch SingleTaskGP, independent outputs, default dimension-scaled RBF priors',
        noise='learned Gaussian observation noise', outcome_transform='Standardize per output',
        input_scaling='unit box', objective_source='all named raw objectives; scalar objective ignored',
        objective_transform='negate minimized objectives for BoTorch maximization',
        acquisition='qLogNoisyExpectedHypervolumeImprovement' if method == 'qlognehvi' else 'qLogNParEGO',
        acquisition_search='maximum over a fresh uniform candidate pool, repaired before scoring',
        acquisition_chunk_size=32, fit_maxiter=100, constraint_eta=.001,
        constraint_handling='enforce the declared repair constraint exactly; model remaining g(x) outputs at g <= 1e-12',
        failed_observations='count attempts and avoid repeats; exclude missing physical outputs from GP',
        reference_point_minimization=reference_point,
        reference_policy=('explicit fixed point' if reference_point is not None else
                          'infer once at first model fit from feasible valid observations, or all valid if none feasible')
                         if method == 'qlognehvi' else None,
        scalarization_weights='seeded Dirichlet(1) draw each model step' if method == 'qlognparego' else None)


def dependencies():
    # Import only when executing this optimizer, not when listing/planning runs.
    try:
        import numpy as np
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        from botorch.acquisition.multi_objective.parego import qLogNParEGO
        from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.multi_objective.hypervolume import infer_reference_point
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise RuntimeError('Install multi-objective BO with: python -m pip install -e ".[mobo]"') from exc
    return locals()


class BoTorchMultiObjectiveSampler(Sampler):
    """CPU q=1 adapters with private random streams and journal-based replay."""
    method = None

    def __init__(self, problem, *, seed, initial, population=4, sigma=.1,
                 bo_initial_points=5, bo_candidates=2048, mobo_mc_samples=128,
                 mobo_reference_point=None):
        self.settings = mobo_settings(self.method, bo_initial_points, bo_candidates,
                                      mobo_mc_samples, mobo_reference_point)
        self.lib = dependencies()
        self.np, self.torch = self.lib['np'], self.lib['torch']
        np = self.np
        self.problem, self.initial = problem, list(problem.validate_design(initial))
        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.span = np.asarray(problem.upper_bounds, dtype=float)-self.lower
        if not np.all(np.isfinite(self.span)) or np.any(self.span <= 0):
            raise ValueError('Multi-objective BO requires finite nonzero real-variable bounds')
        self.repair = CandidateRepair(problem)
        self.repair.check_initial(self.initial)
        self.rng = np.random.RandomState(seed)
        count = bo_initial_points-1
        self.warmup = (np.column_stack([self.rng.permutation(count) for _ in range(problem.dimension)])
                       + self.rng.random_sample((count, problem.dimension)))/count
        self.space = ObjectiveSpace()
        self.modeled_constraint_names = ()
        self.seen, self.x, self.y, self.feasible = [], [], [], []
        self.pending = self.model = self.acquisition = None
        self.reference_point = self.weights = None
        self.proposal_metadata = None

    def _observe(self, x, result):
        normalized = (self.np.asarray(self.problem.validate_design(x))-self.lower)/self.span
        invalid = (result.get('status') in ('failed', 'penalized')
                   or not result.get('metadata', {}).get('valid_cfd', True))
        if not invalid:
            objectives = result.get('objectives')
            if not isinstance(objectives, dict) or len(objectives) < 2:
                raise ValueError('Multi-objective BO requires at least two named physical objectives; no scalar fallback')
            reference = self.settings['reference_point_minimization']
            if reference is not None and set(reference) != set(objectives):
                raise ValueError('Reference point names must exactly match the physical objective names')
            fitness = self.space.fitness(result)
            if not math.isfinite(fitness.violation):
                raise ValueError('Infeasibility must be represented by finite named constraints')
            repaired = self.repair.info.get('constraint') if self.repair.info is not None else None
            if repaired in self.space.constraint_names and result['constraints'][repaired] > 1e-12:
                raise ValueError('The evaluated design violates its declared candidate-repair constraint')
            # P003 volume is known from geometry and enforced before acquisition
            # scoring. Learning its nearly constant zero slack would invent noise
            # and incorrectly discount feasible candidates. Other constraints
            # remain GP outcomes, including infeasible but valid CFD observations.
            self.modeled_constraint_names = tuple(k for k in self.space.constraint_names if k != repaired)
            self.x.append(normalized.tolist())
            self.y.append([-v for v in fitness.objectives]
                          + [float(result['constraints'][k]) for k in self.modeled_constraint_names])
            self.feasible.append(fitness.violation == 0)
        self.seen.append(normalized.tolist())

    def initialize(self, x, result):
        if self.seen or list(x) != self.initial:
            raise ValueError('Initialize multi-objective BO once with its baseline design')
        self._observe(x, result)

    def _novel(self, points):
        mask = self.np.ones(len(points), dtype=bool)
        for previous in self.seen:
            mask &= self.np.max(self.np.abs(points-previous), axis=1) > 1e-9
        return mask

    def _fit(self):
        torch, lib = self.torch, self.lib
        self.train_x = torch.tensor(self.x, dtype=torch.double, device='cpu')
        self.train_y = torch.tensor(self.y, dtype=torch.double, device='cpu')
        self.model = lib['SingleTaskGP'](self.train_x, self.train_y,
            outcome_transform=lib['Standardize'](m=self.train_y.shape[-1]))
        mll = lib['ExactMarginalLogLikelihood'](self.model.likelihood, self.model)
        lib['fit_gpytorch_mll'](mll, optimizer_kwargs={'options': {'maxiter': self.settings['fit_maxiter']}})
        self.model.eval()

    def _acquisition(self):
        torch, lib = self.torch, self.lib
        count = len(self.space.names)
        objective = lib['IdentityMCMultiOutputObjective'](outcomes=list(range(count)))
        constraints = [(lambda samples, index=i: samples[..., index]-1e-12)
                       for i in range(count, self.train_y.shape[-1])]
        sampler = lib['SobolQMCNormalSampler'](sample_shape=torch.Size([self.settings['mc_samples']]),
                                              seed=int(self.rng.randint(2**31)))
        common = dict(model=self.model, X_baseline=self.train_x, objective=objective,
                      constraints=constraints or None, sampler=sampler, prune_baseline=False,
                      eta=self.settings['constraint_eta'])
        if self.method == 'qlognehvi':
            if self.reference_point is None:
                fixed = self.settings['reference_point_minimization']
                if fixed is not None:
                    self.reference_point = torch.tensor([-fixed[k] for k in self.space.names], dtype=torch.double)
                else:
                    values = self.train_y[:, :count]
                    if any(self.feasible):
                        values = values[torch.tensor(self.feasible, dtype=torch.bool)]
                    self.reference_point = lib['infer_reference_point'](values)
            return lib['qLogNoisyExpectedHypervolumeImprovement'](ref_point=self.reference_point, **common)
        self.weights = self.rng.dirichlet(self.np.ones(count)).tolist()
        return lib['qLogNParEGO'](scalarization_weights=torch.tensor(self.weights, dtype=torch.double), **common)

    def _score(self, points):
        torch = self.torch
        values = []
        with torch.no_grad():
            for batch in torch.tensor(points, dtype=torch.double).split(self.settings['acquisition_chunk_size']):
                values.append(self.acquisition(batch.unsqueeze(-2)))
        scores = torch.cat(values).cpu().numpy()
        if not self.np.all(self.np.isfinite(scores)):
            raise RuntimeError('BoTorch returned nonfinite acquisition values')
        return scores

    def _model_proposal(self):
        torch, np = self.torch, self.np
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.settings['torch_threads'])
            # Preserve the caller's CPU RNG. No GPU model or GPU RNG is used.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(self.rng.randint(2**31)))
                self._fit()
                self.acquisition = self._acquisition()
                originals = self.rng.random_sample((self.settings['candidates'], self.problem.dimension))
                pool = self.repair.apply(originals)
                novel = self._novel(pool)
                pool, originals = pool[novel], originals[novel]
                unique = self.repair.distinct(pool)
                pool, originals = pool[unique], originals[unique]
                if not len(pool):
                    return None
                index = int(np.argmax(self._score(pool)))
                return pool[index], originals[index]
        finally:
            torch.set_num_threads(original_threads)

    @property
    def diagnostics(self):
        reference = (dict(zip(self.space.names, (-self.reference_point).tolist()))
                     if self.reference_point is not None else None)
        return dict(mobo_valid_observations=len(self.x), mobo_excluded_attempts=len(self.seen)-len(self.x),
                    mobo_objective_names=list(self.space.names or ()),
                    mobo_constraint_names=list(self.space.constraint_names or ()),
                    mobo_modeled_constraint_names=list(self.modeled_constraint_names),
                    mobo_acquisition_reference_point=reference)

    def ask(self):
        if not self.seen:
            raise RuntimeError('Initialize multi-objective BO before ask')
        if self.pending is not None:
            raise RuntimeError('tell must complete the pending evaluation before ask')
        np = self.np
        model_step = len(self.seen) >= self.settings['initial_points'] and len(self.x) >= 2
        if model_step:
            candidate = self._model_proposal()
            if candidate is None:
                self.stop_reason = 'no_new_design_found'
                return []
            point, proposed = candidate
        else:
            index = len(self.seen)-1
            proposed = self.warmup[index] if index < len(self.warmup) else self.rng.random_sample(self.problem.dimension)
            for _ in range(1000):
                point = self.repair.apply(np.asarray([proposed]))[0]
                if self._novel(np.asarray([point]))[0]:
                    break
                proposed = self.rng.random_sample(self.problem.dimension)
            else:
                self.stop_reason = 'no_new_design_found'
                return []
        design = np.clip(self.lower+self.span*point, self.lower, self.lower+self.span).tolist()
        self.pending = [design]
        metadata = self.repair.metadata(proposed, point) or {}
        metadata['mobo'] = dict(method=self.method, phase='acquisition' if model_step else 'initialization',
            **self.diagnostics, scalarization_weights=(dict(zip(self.space.names, self.weights))
                                                       if model_step and self.weights is not None else None))
        self.proposal_metadata = metadata
        return [list(design)]

    def tell(self, designs, results):
        if self.pending is None or designs != self.pending or len(results) != 1:
            raise ValueError('Multi-objective BO feedback must match the pending design')
        self._observe(designs[0], results[0])
        self.pending = None


class QLogNEHVISampler(BoTorchMultiObjectiveSampler):
    method = 'qlognehvi'


class QLogNParEGOSampler(BoTorchMultiObjectiveSampler):
    method = 'qlognparego'

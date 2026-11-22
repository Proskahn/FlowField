"""CMA-ES proposals in normalized coordinates with optional evaluation repair."""

from .samplers import Sampler


class CMAESSampler(Sampler):
    def __init__(self, problem, *, seed, initial, population=4, sigma=.1):
        try:
            import cma
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install CMA-ES with: python -m pip install -e ".[optim]"') from exc
        self.np = np
        self.lower = np.asarray(problem.lower_bounds)
        self.span = np.asarray(problem.upper_bounds) - self.lower
        from .repair import evaluation_repair
        self.evaluation_repair = evaluation_repair(problem, initial)
        normalized = (np.asarray(initial) - self.lower) / self.span
        self.es = cma.CMAEvolutionStrategy(normalized.tolist(), sigma, dict(
            bounds=[0, 1], popsize=population, CMA_diagonal=problem.dimension > 100,
            CMA_mirrors=0, randn=np.random.RandomState(seed).randn, seed=float('nan'),
            verbose=-9, verb_log=0, verb_disp=0))
        # Shall we use the diagonal or not? We do have higher dimensional problem(>2000, )
        self.pending = None
        self.diagnostics = {}

    def ask(self):
        if self.pending is not None:
            raise RuntimeError('tell must complete the previous CMA-ES batch before ask')
        stopped = self.es.stop()
        if stopped:
            self.stop_reason = 'cma_termination'
            self.diagnostics = {'cma_stop': {str(k): str(v) for k, v in stopped.items()}}
            return []
        self.pending = self.es.ask()
        self.generation = int(self.es.countiter) + 1
        self.designs = [(self.lower + self.span * self.np.asarray(x)).tolist() for x in self.pending]
        if self.evaluation_repair is not None:
            # Keep the sampled search points for pycma's distribution update.
            # Their fitness is the composed objective f(repair(x)).
            self.designs, self.batch_proposal_metadata = self.evaluation_repair.prepare_batch(self.designs)
        return self.designs

    def tell(self, designs, results):
        if self.pending is None or designs != self.designs or len(results) != len(designs):
            raise ValueError('CMA-ES feedback must match the complete pending batch')
        self.es.tell(self.pending, [r['objective'] for r in results])
        self.pending = None

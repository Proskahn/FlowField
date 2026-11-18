"""Pycma lq-CMA-ES with serial, strictly budgeted true evaluations.

SurrogatePopulation owns the model and adaptive evaluation decisions, as in
cma.fmin_lq_surr2. Its synchronous fitness callback is bridged to ask/tell by
replaying only the current population from a model snapshot. Cached scalar
observations satisfy replay; neither this adapter nor replay executes CFD.
"""

from copy import deepcopy
import math

from .samplers import CMAESSampler


def lq_settings():
    return dict(implementation='pycma SurrogatePopulation + CMAEvolutionStrategy',
        model='LQModel: linear, diagonal quadratic, full quadratic',
        tau_truth_threshold=.85, min_evals_percent=2, model_max_size_factor=1,
        inject_model_optimum=True, restarts=0, baseline_in_model=True,
        coordinate_system='normalized bounds [0,1]',
        objective_source='observed scalar objective, including explicit failure penalties',
        budget_source='true evaluations only, including baseline and failures',
        proposal_recovery='deterministic current-population replay; no repeated CFD')


class _NeedEvaluation(Exception):
    def __init__(self, x):
        self.x = x.copy()


def _unavailable_fitness(x):
    raise RuntimeError('lq-CMA-ES fitness is supplied through ask/tell only')


class LQCMAESSampler(CMAESSampler):
    """Adaptive single-evaluation requests within actual CMA generations.

Only true results are returned to the runner. Surrogate scores stay inside
pycma's distribution update and never become logged observations or incumbents.
"""
    manages_generations = True

    def __init__(self, problem, *, seed, initial, population=4, sigma=.1):
        super().__init__(problem, seed=seed, initial=initial, population=population, sigma=sigma)
        from cma.fitness_models import SurrogatePopulation
        self.surrogate = SurrogatePopulation(_unavailable_fitness)
        self.surrogate.settings.logging_push = False
        self.population_size = population
        self._population = None
        self._observations = []
        self._next = None
        self.true_evaluations = 0
        self.surrogate_only = 0
        self._refresh()

    def _refresh(self):
        self.diagnostics.update(complete_generations=int(self.es.countiter),
            partial_generation_evaluations=len(self._observations),
            lq_true_evaluations=self.true_evaluations,
            lq_surrogate_only_evaluations=self.surrogate_only,
            lq_completed_population_members=int(self.es.countiter)*self.population_size,
            lq_model_size=int(self.surrogate.model.size),
            lq_model_type='+'.join(self.surrogate.model.types) or 'linear')

    @staticmethod
    def _value(result):
        value = float(result['objective'])
        if not math.isfinite(value):
            raise ValueError('lq-CMA-ES requires finite observed objectives')
        return value

    def initialize(self, x, result):
        if self.true_evaluations:
            raise RuntimeError('lq-CMA-ES baseline is already initialized')
        z = (self.np.asarray(x)-self.lower)/self.span
        self.surrogate.model.add_data_row(z, self._value(result))
        self.surrogate.evaluations = self.true_evaluations = 1
        self.es.countevals = 1
        self._refresh()

    def _advance(self):
        """Run pycma to its next unknown fitness or commit a complete population."""
        trial = deepcopy(self.surrogate)
        cursor = 0

        def observed(z):
            nonlocal cursor
            if cursor == len(self._observations):
                raise _NeedEvaluation(z)
            old, value = self._observations[cursor]
            if not self.np.array_equal(z, old):
                raise RuntimeError('lq-CMA-ES population replay changed an evaluated proposal')
            cursor += 1
            return value

        trial.fitness = observed
        try:
            scores = trial(self._population)
        except _NeedEvaluation as request:
            self._next = request.x
            return
        if cursor != len(self._observations) or not self.np.isfinite(scores).all():
            raise RuntimeError('Invalid lq-CMA-ES population feedback')
        trial.fitness = _unavailable_fitness  # Do not retain the replay closure.
        self.es.tell(self._population, scores)
        self.es.countevals = self.true_evaluations
        self.es.inject([trial.model.xopt])  # Same model-optimum injection as fmin_lq_surr2.
        self.surrogate_only += len(self._population)-len(self._observations)
        self.surrogate = trial
        self._population, self._next, self._observations = None, None, []

    def ask(self):
        if not self.true_evaluations:
            raise RuntimeError('Initialize lq-CMA-ES with the observed baseline first')
        if self.pending is not None:
            raise RuntimeError('tell must complete the pending lq-CMA-ES evaluation before ask')
        if self._population is None:
            stopped = self.es.stop()
            if stopped:
                self.stop_reason = 'cma_termination'
                self.diagnostics['cma_stop'] = {str(k):str(v) for k,v in stopped.items()}
                return []
            self._population = self.es.ask()
            self.generation = int(self.es.countiter)+1
            self._advance()
        if self._next is None:
            raise RuntimeError('Pycma did not request a true evaluation in its new population')
        self.pending = (self.lower+self.span*self._next).tolist()
        self.proposal_metadata = dict(source='pycma_lq_true_evaluation',
            population_size=self.population_size,
            true_evaluation_in_generation=len(self._observations)+1)
        self._refresh()
        return [list(self.pending)]

    def tell(self, designs, results):
        if self.pending is None or designs != [self.pending] or len(results) != 1:
            raise ValueError('lq-CMA-ES feedback must match its pending single evaluation')
        value = self._value(results[0])
        self._observations.append((self._next.copy(), value))
        self.true_evaluations += 1
        self.pending = None
        self._advance()
        self._refresh()

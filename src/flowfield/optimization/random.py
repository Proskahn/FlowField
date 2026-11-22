"""Seeded random search with optional problem-owned feasibility repair."""

import random

from .samplers import Sampler


def random_settings(problem):
    """Record the proposal distribution separately from feasibility repair."""
    repair = getattr(problem, 'candidate_repair', None)
    return dict(proposal_source='problem.sample_design' if hasattr(problem, 'sample_design')
                else 'independent uniform coordinates over declared bounds',
                candidate_repair=repair,
                sampling_policy='repair proposals before duplicate checks and evaluation' if repair is not None
                else 'evaluate unique proposals without repair',
                rng='private Python random.Random',
                duplicate_policy='exclude baseline and all previously proposed evaluation designs',
                maximum_draws_per_candidate=1000, objective_adaptation=False)


class RandomSampler(Sampler):
    """Seeded random proposals with optional problem-owned feasibility repair.

    Repaired uniform draws are not uniform over the feasible domain. Feedback
    never changes the proposal distribution; failed designs also remain seen.
    """
    def __init__(self, problem, *, seed, initial, **options):
        self.problem = problem
        self.rng = random.Random(seed)
        self.seen = {tuple(initial)}
        self.repair = None
        if getattr(problem, 'candidate_repair', None) is not None:
            from .repair import CandidateRepair
            try:
                self.repair = CandidateRepair(problem)
                self.repair.check_initial(initial)
            except ImportError as exc:
                raise RuntimeError('Install random-search repair dependencies with: '
                                   'python -m pip install -e ".[random]"') from exc
        self.proposal_metadata = None
        self.diagnostics = dict(random_draws=0, random_repaired_draws=0, random_duplicate_draws=0)

    def ask(self):
        self.proposal_metadata = None
        for _ in range(1000):
            x = (list(self.problem.sample_design(self.rng)) if hasattr(self.problem, 'sample_design') else
                 [self.rng.uniform(lo, hi) for lo, hi in zip(self.problem.lower_bounds, self.problem.upper_bounds)])
            self.diagnostics['random_draws'] += 1
            metadata = None
            if self.repair is not None:
                repair = self.repair
                proposed = (repair.np.asarray(x)-repair.lower)/repair.span
                evaluated = repair.apply(proposed[None, :])[0]
                metadata = repair.metadata(proposed, evaluated)
                metadata['proposed_x'] = list(x)
                if metadata['applied']:
                    self.diagnostics['random_repaired_draws'] += 1
                    x = (repair.lower+repair.span*evaluated).tolist()
            if tuple(x) not in self.seen:
                self.seen.add(tuple(x))
                self.proposal_metadata = metadata
                return [x]
            self.diagnostics['random_duplicate_draws'] += 1
        self.stop_reason = 'no_new_design_found'
        return []

    def tell(self, designs, results):
        pass

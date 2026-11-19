"""Optional problem-owned feasibility repair for optimizer adapters.

The hook accepts physical design rows and a componentwise lower anchor. It
must leave feasible rows unchanged and return rows on the anchor-to-input
segments. Samplers apply it BEFORE novelty checks and acquisition scoring;
Models may instead explicitly learn the composed objective f(repair(x)) in
search coordinates (CMA-ES, lq-CMA-ES and COBYLA). Journals always identify the
actual evaluated coordinates and retain the original proposals.
"""


EVALUATION_REPAIR_OPTIMIZERS = ('cmaes', 'lqcmaes', 'nsga2', 'cobyla')


def evaluation_repair_settings(problem, optimizer):
    """Opt in to repair in the remaining adapters without changing other problems."""
    if optimizer not in EVALUATION_REPAIR_OPTIMIZERS or not getattr(problem, 'repair_before_evaluation', False):
        return {}
    return dict(candidate_repair=problem.candidate_repair,
                repair_stage='before CFD',
                repair_feedback=('population uses repaired design coordinates' if optimizer == 'nsga2'
                                 else 'search coordinates optimize f(repair(x))'))


def evaluation_repair(problem, initial):
    if not getattr(problem, 'repair_before_evaluation', False):
        return None
    try:
        repair = CandidateRepair(problem)
        if repair.info is None:
            raise ValueError('repair_before_evaluation requires candidate_repair metadata')
        repair.check_initial(initial)
        return repair
    except ImportError as exc:
        raise RuntimeError('Install volume-repair dependencies with: '
                           'python -m pip install -e ".[p003]"') from exc


class CandidateRepair:
    def __init__(self, problem):
        import numpy as np
        self.np = np
        self.problem = problem
        self.info = getattr(problem, 'candidate_repair', None)
        self.lower = np.asarray(problem.lower_bounds)
        self.span = np.asarray(problem.upper_bounds) - self.lower
        if self.info is not None and not callable(getattr(problem, 'repair_candidates', None)):
            raise TypeError('candidate_repair metadata requires a repair_candidates hook')

    def apply(self, candidates, *, anchor=None):
        if self.info is None:
            return candidates
        np = self.np
        candidates = np.asarray(candidates, dtype=float)
        origin = self.lower if anchor is None else self.lower+self.span*np.asarray(anchor)
        physical = self.lower+self.span*candidates
        repaired = np.asarray(self.problem.repair_candidates(physical, anchor=origin), dtype=float)
        if (repaired.shape != physical.shape or not np.all(np.isfinite(repaired))
                or np.any(repaired < origin) or np.any(repaired > physical)):
            raise ValueError('Candidate repair returned invalid rows or escaped the anchor-to-input bounds')
        # Preserve exact coordinates for unchanged rows (and replay stability).
        changed = np.any(repaired != physical, axis=1)
        output = candidates.copy()
        output[changed] = (repaired[changed]-self.lower)/self.span
        return output

    def check_initial(self, initial):
        if self.info is not None:
            normalized = (self.np.asarray(initial)-self.lower)/self.span
            if not self.np.array_equal(self.apply(normalized[None, :])[0], normalized):
                raise ValueError('Initial design violates candidate constraints; supply a feasible baseline')

    def prepare_batch(self, designs):
        """Return physical evaluated designs and per-design proposal metadata."""
        original = self.np.asarray(designs, dtype=float)
        normalized = (original-self.lower)/self.span
        repaired = self.apply(normalized)
        changed = self.np.any(repaired != normalized, axis=1)
        physical = original.copy()
        physical[changed] = self.lower+self.span*repaired[changed]
        metadata = []
        for raw, proposed, evaluated in zip(original, normalized, repaired):
            info = self.metadata(proposed, evaluated)
            if info is not None:
                info['proposed_x'] = raw.tolist()
            metadata.append(info)
        return physical.tolist(), metadata

    def distinct(self, candidates):
        """Stable indices: repair may map several proposals to the same design."""
        if self.info is None:
            return self.np.arange(len(candidates))
        return self.np.sort(self.np.unique(candidates, axis=0, return_index=True)[1])

    def metadata(self, proposed, evaluated, *, anchor=None):
        if self.info is None:
            return None
        return dict(candidate_repair=self.info,
                    applied=bool(self.np.any(proposed != evaluated)),
                    proposed_x=(self.lower+self.span*proposed).tolist(),
                    anchor_x=(self.lower if anchor is None else self.lower+self.span*anchor).tolist())

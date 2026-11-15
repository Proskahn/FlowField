"""Optional problem-owned feasibility repair for model-based samplers.

The hook accepts physical design rows and a componentwise lower anchor. It
must leave feasible rows unchanged and return rows on the anchor-to-input
segments. Samplers apply it BEFORE novelty checks and acquisition scoring;
training, journals and CFD then use the same repaired design coordinates.
"""


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

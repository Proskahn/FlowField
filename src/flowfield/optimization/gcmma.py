"""Conservative GCMMA steps with supplied objective and constraint derivatives.

Uses the optional mmapy implementation of Svanberg's subproblem solver. This
module does not approximate derivatives, evaluate CFD, or repair trial designs.
"""

from dataclasses import dataclass
import math


def dependencies():
    try:
        import numpy as np
        import mmapy
    except ImportError as exc:
        raise RuntimeError('Install GCMMA with: python -m pip install -e ".[gcmma]"') from exc
    return np, mmapy


@dataclass(frozen=True)
class GCMMAOptions:
    move: float = .01
    max_inner: int = 15
    conservativeness_tolerance: float = 1e-8
    subproblem_tolerance: float = 1e-9
    constraint_tolerance: float = 1e-8
    asymptote_initial: float = .5
    asymptote_decrease: float = .7
    asymptote_increase: float = 1.43
    minimum_regularization: float = 1e-5
    slack_penalty: float = 1000.

    def __post_init__(self):
        if type(self.max_inner) is not int or self.max_inner < 1:
            raise ValueError('max_inner must be a positive integer')
        for name, value in vars(self).items():
            if name == 'max_inner':
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.move > 1 or self.asymptote_decrease >= 1 or self.asymptote_increase <= 1:
            raise ValueError('Require move <= 1, asymptote_decrease < 1 and asymptote_increase > 1')


class GCMMA:
    """Unit-box GCMMA with conservative inner iterations and hard acceptance.

    Call ``propose(f, df, g, dg)`` at the current ``x``, then evaluate ``trial``.
    ``observe(f_trial, g_trial)`` returns True for an accepted iterate. Otherwise
    it creates a new ``trial`` with stronger conservative approximations, using
    the same derivatives. An exhausted inner loop raises rather than accepting
    a nonconservative or infeasible design. Constraints use g <= 0.
    """

    def __init__(self, initial, constraints=1, *, options=None):
        self.np, self.mma = dependencies()
        np = self.np
        self.options = options or GCMMAOptions()
        self.x = np.asarray(initial, dtype=float).reshape(-1, 1)
        if not self.x.size or not np.all(np.isfinite(self.x)) or np.any(self.x < 0) or np.any(self.x > 1):
            raise ValueError('GCMMA initial design must lie in the finite unit box')
        if type(constraints) is not int or constraints < 1:
            raise ValueError('GCMMA requires at least one inequality constraint')
        self.n, self.m = self.x.size, constraints
        self.xold1, self.xold2 = self.x.copy(), self.x.copy()
        self.low, self.upp = self.x.copy(), self.x.copy()
        self.iteration, self.inner = 0, 0
        self.pending = False
        self.raa0 = np.array([[self.options.minimum_regularization]])
        self.raa = np.full((self.m, 1), self.options.minimum_regularization)

    def _array(self, values, shape, name):
        array = self.np.asarray(values, dtype=float)
        if array.size != math.prod(shape) or not self.np.all(self.np.isfinite(array)):
            raise ValueError(f'{name} must have shape {shape} and finite values')
        return array.reshape(shape)

    def propose(self, objective, gradient, constraints, jacobian):
        if self.pending:
            raise RuntimeError('Observe the pending GCMMA trial first')
        np, opt = self.np, self.options
        self.f = self._array(objective, (1, 1), 'objective')
        self.df = self._array(gradient, (self.n, 1), 'objective gradient')
        self.g = self._array(constraints, (self.m, 1), 'constraints')
        self.dg = self._array(jacobian, (self.m, self.n), 'constraint Jacobian')
        self.iteration += 1
        self.inner = 0
        self.low, self.upp, self.raa0, self.raa = self.mma.asymp(
            self.iteration, self.n, self.x, self.xold1, self.xold2,
            np.zeros_like(self.x), np.ones_like(self.x), self.low, self.upp,
            self.raa0, self.raa, opt.minimum_regularization,
            opt.minimum_regularization, self.df, self.dg,
            asyinit=opt.asymptote_initial, asydecr=opt.asymptote_decrease,
            asyincr=opt.asymptote_increase)
        # Explicit move bounds, held fixed throughout this outer iteration.
        self.lower = np.maximum(0., self.x-opt.move)
        self.upper = np.minimum(1., self.x+opt.move)
        self.pending = True
        self._solve()
        return self.trial.copy().ravel()

    def _solve(self):
        np, opt = self.np, self.options
        result = self.mma.gcmmasub(
            self.m, self.n, self.iteration, opt.subproblem_tolerance,
            self.x, self.lower, self.upper, self.low, self.upp,
            self.raa0, self.raa, self.f, self.df, self.g, self.dg,
            1., np.zeros((self.m, 1)), np.full((self.m, 1), opt.slack_penalty),
            np.zeros((self.m, 1)))
        self.trial, self.f_approx, self.g_approx = result[0], result[-2], result[-1]
        if (not np.all(np.isfinite(self.trial)) or np.any(self.trial < self.lower)
                or np.any(self.trial > self.upper)):
            raise RuntimeError('GCMMA subproblem returned a nonfinite or out-of-bounds trial')

    def observe(self, objective, constraints):
        if not self.pending:
            raise RuntimeError('No pending GCMMA trial')
        opt = self.options
        f = self._array(objective, (1, 1), 'trial objective')
        g = self._array(constraints, (self.m, 1), 'trial constraints')
        conservative = self.mma.concheck(self.m, opt.conservativeness_tolerance,
                                         self.f_approx, f, self.g_approx, g)
        if conservative and self.np.all(g <= opt.constraint_tolerance):
            self.xold2, self.xold1 = self.xold1.copy(), self.x.copy()
            self.x = self.trial.copy()
            self.pending = False
            return True
        self.inner += 1
        if self.inner >= opt.max_inner:
            raise RuntimeError('GCMMA inner iterations exhausted without a conservative, feasible step; '
                               'check derivative accuracy or reduce the move limit')
        self.raa0, self.raa = self.mma.raaupdate(
            self.trial, self.x, self.lower, self.upper, self.low, self.upp,
            f, g, self.f_approx, self.g_approx, self.raa0, self.raa,
            opt.minimum_regularization, opt.minimum_regularization,
            opt.conservativeness_tolerance)
        if conservative:
            # A soft subproblem slack is not permission to violate the real
            # constraint. Reduce the move box before trying another candidate.
            self.lower = (self.lower+self.x)/2
            self.upper = (self.upper+self.x)/2
        self._solve()
        return False

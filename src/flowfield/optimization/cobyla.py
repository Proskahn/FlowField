"""Deterministic SciPy COBYLA adapter for the shared ask/tell runner.

SciPy owns the constrained trust-region algorithm. Replaying cheap optimizer
steps to the next unknown objective avoids threads and never repeats CFD.
The runner's existing journals therefore support exact interruption/resume.
"""
import math
from .samplers import Sampler


def dependencies():
    try:
        import scipy
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("Install COBYLA with Python 3.11+: python -m pip install -e '.[p006]'") from exc
    if tuple(int(v) for v in scipy.__version__.split('.')[:2]) < (1,16):
        raise RuntimeError('This COBYLA adapter requires SciPy >=1.16 (the PRIMA implementation)')
    return minimize


class _NeedEvaluation(Exception):
    def __init__(self,x):self.x=x


class _ObjectiveConverged(Exception):
    pass


class COBYLASampler(Sampler):
    def __init__(self,problem,*,initial=None,seed=42,population=4,sigma=.15):
        self.minimize=dependencies()
        self.problem=problem
        self.initial=list(problem.initial_design() if initial is None else initial)
        self.observations=[]
        self.pending=None
        self.done=False
        self.stop_reason=None
        self.lower=list(problem.lower_bounds)
        self.scale=[hi-lo for lo,hi in zip(self.lower,problem.upper_bounds)]

    def _normalized(self,x):return [(v-lo)/d for v,lo,d in zip(x,self.lower,self.scale)]
    def _physical(self,x):return [lo+v*d for v,lo,d in zip(x,self.lower,self.scale)]

    def initialize(self,x,result):
        if self.observations:raise ValueError('COBYLA already initialized')
        self.initial=list(x)
        self.observations.append((list(x),result))

    def ask(self):
        if self.done:return []
        if self.pending is not None:raise RuntimeError('tell is required before the next ask')
        previous=[]
        evaluations=0

        def constraints(z):
            bounded=[min(1.,max(0.,float(v))) for v in z]
            values=[float(v) for v in z]+[1-float(v) for v in z]
            if hasattr(self.problem,'geometric_constraints'):
                values.extend(-float(v) for v in self.problem.geometric_constraints(self._physical(bounded)).values())
            return values

        def objective(z):
            nonlocal evaluations
            # COBYLA can evaluate outside explicit bounds. Keep CFD within its domain.
            if any(v<0 or v>1 for v in z):
                return 1e12+1e6*sum(max(-v,0)+max(v-1,0) for v in z)
            x=self._physical(z)
            known=next((r for old,r in self.observations if max(abs(a-b) for a,b in zip(x,old))<1e-12),None)
            if known is None:raise _NeedEvaluation(x)
            evaluations+=1
            value=float(known.get('unpenalized_objective',known['objective']))
            if not math.isfinite(value):raise ValueError('COBYLA requires finite objective values')
            feasible=known.get('status')=='ok' and all(v<=1e-12 for v in known.get('constraints',{}).values())
            # Paper's relative objective stopping rule is applied to distinct,
            # successive feasible evaluations after the initial simplex is built.
            if feasible and previous and evaluations>self.problem.dimension+1:
                px,pv=previous[-1]
                if max(abs(a-b) for a,b in zip(x,px))>1e-10 and abs(value-pv)/max(abs(pv),1e-12)<1e-5:
                    raise _ObjectiveConverged
            previous[:]=[(x,value)] if feasible else []
            return value

        try:
            result=self.minimize(objective,self._normalized(self.initial),method='COBYLA',
                constraints=[dict(type='ineq',fun=constraints)],
                options=dict(rhobeg=.1,tol=1e-5,catol=1e-8,maxiter=10000))
            self.stop_reason=str(result.message)
        except _NeedEvaluation as request:
            self.pending=request.x
            return [list(request.x)]
        except _ObjectiveConverged:
            self.stop_reason='Paper relative objective change < 1e-5'
        self.done=True
        return []

    def tell(self,designs,results):
        if len(designs)!=1 or len(results)!=1 or self.pending is None:
            raise ValueError('COBYLA expects exactly its pending single evaluation')
        if list(designs[0])!=self.pending:raise ValueError('Design differs from pending COBYLA proposal')
        self.observations.append((list(designs[0]),results[0]))
        self.pending=None

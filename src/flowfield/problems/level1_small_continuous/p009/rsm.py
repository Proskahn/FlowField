"""Full quadratic RSM and a reproducible real-valued GA on its fixed surface."""
import itertools
from .config import BOUNDS, PARAMETERS
from .reference import TABLE5, PRODUCT_BASELINE, REPORTED_OPTIMUM, coded, printed_equation

TERMS = ('1', 'A', 'B', 'C', 'A2', 'B2', 'C2', 'AB', 'AC', 'BC')


def features(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, 3)
    a, b, c = x.T
    return np.column_stack((np.ones(len(x)), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c))


def fit(rows, responses):
    import numpy as np
    x = np.array([coded(row) for row in rows]); y = np.asarray(responses, dtype=float)
    if x.shape != (len(y), 3) or len(y) <= len(TERMS) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('RSM needs more than ten complete finite observations')
    if np.max(np.abs(x)) > 1+1e-9:
        raise ValueError('Do not fit out-of-domain baselines into the Table 4 surface')
    design = features(x)
    coef, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    if rank != len(TERMS):
        raise ValueError('RSM design matrix is rank deficient')
    residual = y-design@coef
    sse, sst = float(residual@residual), float(np.sum((y-y.mean())**2))
    if sst <= 1e-30:
        raise ValueError('No response variation in RSM data')
    # Leave all duplicate designs out together: repeated deterministic CFD is
    # not independent validation. Ordinary LOO would leak identical designs.
    holdout = []
    for key in dict.fromkeys(tuple(row) for row in rows):
        take = np.array([tuple(row) == key for row in rows])
        train = design[~take]
        beta, _, r, _ = np.linalg.lstsq(train, y[~take], rcond=None)
        if r != len(TERMS):
            holdout = None
            break
        holdout.extend((y[take]-design[take]@beta).tolist())
    press = sum(v*v for v in holdout) if holdout is not None else None
    return dict(terms=list(TERMS), coefficients=coef.tolist(), bounds=[list(b) for b in BOUNDS],
        coding='(physical-midpoint)/half_range', observations=len(y), unique_designs=len(set(map(tuple, rows))),
        rank=int(rank), condition_number=float(singular[0]/singular[-1]), r2=1-sse/sst,
        adjusted_r2=1-(sse/(len(y)-len(TERMS)))/(sst/(len(y)-1)), rmse=(sse/len(y))**.5,
        max_abs_error=float(np.max(np.abs(residual))),
        grouped_holdout_predicted_r2=None if press is None else 1-press/sst,
        validation_note='Duplicate designs held out together; no claim of independent physical validation.')


def predict(model, normalized):
    import numpy as np
    x = np.asarray(normalized, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError('RSM prediction requires three normalized coordinates in [0,1]')
    return float(features(2*x-1)[0] @ np.array(model['coefficients']))


def optimize_ga(model, *, seed=42, population=64, generations=100):
    """Tournament selection, blend crossover, Gaussian mutation and elitism.

    The paper omits GA settings; these settings are declared reconstruction
    choices. CFD qualification of the returned design is mandatory in a study.
    """
    import numpy as np
    if type(population) is not int or population < 8 or type(generations) is not int or generations < 1:
        raise ValueError('GA needs population >= 8 and generations >= 1')
    rng = np.random.default_rng(seed)
    candidates = rng.random((population, 3))
    candidates[:8] = list(itertools.product((0., 1.), repeat=3))
    history = []
    coef = np.array(model['coefficients'])
    for _ in range(generations):
        fitness = features(2*candidates-1)@coef
        order = np.argsort(-fitness)
        history.append(float(fitness[order[0]]))
        next_population = [candidates[order[0]].copy(), candidates[order[1]].copy()]
        while len(next_population) < population:
            parents = []
            for _ in range(2):
                tournament = rng.integers(population, size=3)
                parents.append(candidates[tournament[np.argmax(fitness[tournament])]])
            alpha = rng.uniform(-.25, 1.25, size=3)
            child = alpha*parents[0]+(1-alpha)*parents[1]
            child += (rng.random(3) < 1/3)*rng.normal(0, .1, size=3)
            next_population.append(np.clip(child, 0, 1))
        candidates = np.array(next_population)
    fitness = features(2*candidates-1)@coef
    best = candidates[int(np.argmax(fitness))].tolist()
    return dict(normalized_design=best,
        physical_design={k:lo+(hi-lo)*v for k, v, (lo,hi) in zip(PARAMETERS,best,BOUNDS)},
        predicted_performance=predict(model,best), seed=seed, population=population,
        generations=generations, best_history=history, settings_source='declared reconstruction; paper omits settings')


def reference_audit(seed=42):
    import numpy as np
    rows = [r[:3] for r in TABLE5]; y = np.array([r[3] for r in TABLE5])
    model = fit(rows, y)
    errors = [printed_equation(r)-v for r,v in zip(rows,y)]
    # Dense bounded check of Eq.7 is cheap and independent of the refit GA.
    best = max((printed_equation((a,b,c)), a,b,c)
        for a in (70.,115.) for b in (.3,1.2) for c in np.linspace(1.6,2.24,1001))
    return dict(status='completed', evidence='published table/regression audit; zero CFD evaluations',
        paper_doi='10.1002/ese3.70173', table_rows=len(TABLE5),
        table_refit=model, refit_ga=optimize_ga(model,seed=seed),
        printed_equation=dict(coding_inferred=True, max_table_error=max(abs(v) for v in errors),
            rmse=float(np.sqrt(np.mean(np.square(errors)))),
            reported_design_prediction=printed_equation(REPORTED_OPTIMUM),
            literal_physical_variable_prediction=printed_equation(REPORTED_OPTIMUM,coded_variables=False),
            bounded_grid_best=dict(F=best[0],angle_deg=best[1],flow_lpm=best[2],thickness_mm=best[3])),
        reported=dict(optimum=list(REPORTED_OPTIMUM), optimum_F=.0388,
            product_baseline=list(PRODUCT_BASELINE), baseline_F=.0323,
            improvement_percent=100*(.0388/.0323-1), baseline_outside_table4=True),
        paper_validated=False)

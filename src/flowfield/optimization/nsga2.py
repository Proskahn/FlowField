"""Real-valued NSGA-II: constrained dominance, elitism, SBX and mutation.

All named raw objectives are minimized (maximization objectives must be negated
by the problem). No scalarization is used for selection. Operators follow the
standard rank/crowding NSGA-II scheme; see https://pymoo.org/algorithms/moo/nsga2.html.
Uses a private random.Random stream; optional problem repair may require NumPy/SciPy.
"""

from dataclasses import dataclass
import math
from numbers import Real
import random

from .samplers import Sampler


@dataclass(frozen=True)
class Fitness:
    objectives: tuple
    violation: float = 0.
    valid: bool = True

    def dominates(self, other):
        if self.valid != other.valid:
            return self.valid
        if not self.valid:
            return False
        if self.violation != other.violation:
            return self.violation < other.violation
        if self.violation > 0:
            return False
        return (all(a <= b for a, b in zip(self.objectives, other.objectives))
                and any(a < b for a, b in zip(self.objectives, other.objectives)))


class ObjectiveSpace:
    """Validate a stable named objective/constraint schema across observations."""
    def __init__(self):
        self.names = None
        self.constraint_names = None

    def fitness(self, result):
        if (result.get('status') in ('failed', 'penalized')
                or not result.get('metadata', {}).get('valid_cfd', True)):
            return Fitness((), valid=False)
        objectives = result.get('objectives') or {'objective': result['objective']}
        constraints = result.get('constraints', {})
        for label, values in (('objectives', objectives), ('constraints', constraints)):
            if not isinstance(values, dict) or any(not isinstance(k, str) or not k for k in values):
                raise ValueError(f'NSGA-II {label} must be a dictionary of named numbers')
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                   for v in values.values()):
                raise ValueError(f'NSGA-II {label} must be finite real numbers')
        names, constraint_names = tuple(sorted(objectives)), tuple(sorted(constraints))
        if self.names is not None and (names != self.names or constraint_names != self.constraint_names):
            raise ValueError('NSGA-II objective or constraint names changed during the run')
        self.names, self.constraint_names = names, constraint_names
        violation = sum(float(v) for v in constraints.values() if v > 1e-12)
        if not result.get('feasible', True) and violation == 0:
            violation = math.inf
        return Fitness(tuple(float(objectives[k]) for k in names), violation)


def fronts(fitnesses):
    """Fast non-dominated sorting, returning index fronts in deterministic order."""
    dominates = [[] for _ in fitnesses]
    counts = [0] * len(fitnesses)
    for i, a in enumerate(fitnesses):
        for j in range(i+1, len(fitnesses)):
            b = fitnesses[j]
            if a.dominates(b):
                dominates[i].append(j)
                counts[j] += 1
            elif b.dominates(a):
                dominates[j].append(i)
                counts[i] += 1
    current = [i for i, count in enumerate(counts) if count == 0]
    result = []
    while current:
        result.append(current)
        following = []
        for i in current:
            for j in dominates[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        current = sorted(following)
    return result


def crowding(fitnesses, front):
    distances = {i: 0. for i in front}
    if not front or not fitnesses[front[0]].valid or fitnesses[front[0]].violation > 0:
        return distances
    for axis in range(len(fitnesses[front[0]].objectives)):
        ordered = sorted(front, key=lambda i: fitnesses[i].objectives[axis])
        low, high = (fitnesses[ordered[i]].objectives[axis] for i in (0, -1))
        if low == high:  # Constant objectives must not create arbitrary extremes.
            continue
        distances[ordered[0]] = distances[ordered[-1]] = math.inf
        for previous, current, following in zip(ordered, ordered[1:], ordered[2:]):
            distances[current] += ((fitnesses[following].objectives[axis]
                                    - fitnesses[previous].objectives[axis]) / (high-low))
    return distances


class ParetoArchive:
    """All observed feasible nondominated designs, including partial generations."""
    def __init__(self):
        self.space = ObjectiveSpace()
        self.entries = []

    def add(self, record):
        fitness = self.space.fitness(record['result'])
        if not record['feasible'] or not fitness.valid or fitness.violation > 0:
            return
        if any(f.dominates(fitness) or (f == fitness and r['x'] == record['x'])
               for f, r in self.entries):
            return
        self.entries = [(f, r) for f, r in self.entries if not fitness.dominates(f)]
        self.entries.append((fitness, record))

    def to_dict(self):
        return dict(objective_names=list(self.space.names or ()), objective_direction='minimize',
                    constraint_convention='g(x) <= 0; feasibility tolerance 1e-12',
                    scope='all evaluated feasible designs',
                    solutions=[r for _, r in self.entries])


class NSGAIISampler(Sampler):
    """NSGA-II for bounded real variables; baseline belongs to initial population.

    SBX: pair probability .9, per-variable probability .5, eta=15.
    Polynomial mutation: per-variable probability 1/dimension, eta=20.
    Infeasible solutions rank by summed positive constraint violation; invalid
    CFD ranks last regardless of the scalar failure penalty. sigma is unused.
    """
    def __init__(self, problem, *, seed, initial, population=4, sigma=.1):
        if getattr(problem, 'variable_type', 'real') != 'real':
            raise ValueError('NSGA-II currently supports real variables only')
        if type(population) is not int or population < 4:
            raise ValueError('NSGA-II population must be an integer of at least 4')
        self.lower = list(problem.lower_bounds)
        self.span = [hi-lo for lo, hi in zip(problem.lower_bounds, problem.upper_bounds)]
        self.initial = problem.validate_design(initial)
        from .repair import evaluation_repair
        self.evaluation_repair = evaluation_repair(problem, self.initial)
        self.size = population
        self.rng = random.Random(seed)
        self.space = ObjectiveSpace()
        self.population = []
        self.pending = None
        self.generation = 1
        self.initialized = False

    def initialize(self, x, result):
        if self.initialized or list(x) != self.initial:
            raise ValueError('NSGA-II must initialize once with the declared baseline')
        self.population = [(list(x), self.space.fitness(result))]
        self.initialized = True

    def _selection_keys(self):
        fitnesses = [f for _, f in self.population]
        keys = {}
        for rank, front in enumerate(fronts(fitnesses)):
            distances = crowding(fitnesses, front)
            keys.update({i: (rank, -distances[i]) for i in front})
        return keys

    def _children(self, a, b):
        a = [(v-lo)/span for v, lo, span in zip(a, self.lower, self.span)]
        b = [(v-lo)/span for v, lo, span in zip(b, self.lower, self.span)]
        if self.rng.random() < .9:
            for i, (u, v) in enumerate(zip(a, b)):
                if self.rng.random() > .5 or abs(u-v) <= 1e-14:
                    continue
                low, high = sorted((u, v))
                draw = self.rng.random()
                def beta(distance):
                    alpha = 2 - (1 + 2*distance/(high-low)) ** -16
                    return ((draw*alpha) ** (1/16) if draw <= 1/alpha
                            else (1/(2-draw*alpha)) ** (1/16))
                left = .5 * (low+high-beta(low)*(high-low))
                right = .5 * (low+high+beta(1-high)*(high-low))
                a[i], b[i] = (left, right) if self.rng.random() < .5 else (right, left)
        children = []
        for child in (a, b):
            for i, value in enumerate(child):
                value = min(1., max(0., value))
                if self.rng.random() < 1/len(child):
                    draw = self.rng.random()
                    if draw <= .5:
                        value += (2*draw+(1-2*draw)*(1-value)**21)**(1/21)-1
                    else:
                        value += 1-(2*(1-draw)+2*(draw-.5)*value**21)**(1/21)
                child[i] = min(1., max(0., value))
            children.append([lo+span*v for lo, span, v in zip(self.lower, self.span, child)])
        return children

    def ask(self):
        if not self.initialized or self.pending is not None:
            raise RuntimeError('Initialize NSGA-II and tell the complete pending batch before ask')
        if len(self.population) < self.size:
            designs = [[lo+span*self.rng.random() for lo, span in zip(self.lower, self.span)]
                       for _ in range(self.size-len(self.population))]
        else:
            keys = self._selection_keys()
            def parent():
                a, b = self.rng.sample(range(self.size), 2)
                winner = a if keys[a] < keys[b] else b if keys[b] < keys[a] else self.rng.choice((a, b))
                return self.population[winner][0]
            designs = []
            while len(designs) < self.size:
                designs.extend(self._children(parent(), parent()))
            designs = designs[:self.size]
        if self.evaluation_repair is not None:
            designs, self.batch_proposal_metadata = self.evaluation_repair.prepare_batch(designs)
        self.pending = [list(x) for x in designs]
        return designs

    def tell(self, designs, results):
        if self.pending is None or designs != self.pending or len(results) != len(designs):
            raise ValueError('NSGA-II feedback must match the complete pending batch')
        combined = self.population + [(list(x), self.space.fitness(r)) for x, r in zip(designs, results)]
        fitnesses = [f for _, f in combined]
        selected = []
        for front in fronts(fitnesses):
            if len(selected)+len(front) > self.size:
                distances = crowding(fitnesses, front)
                # Seeded tie-breaking prevents permanent preference for old parents.
                self.rng.shuffle(front)
                front.sort(key=lambda i: distances[i], reverse=True)
            selected.extend(front[:self.size-len(selected)])
            if len(selected) == self.size:
                break
        self.population = [combined[i] for i in selected]
        self.pending = None
        self.generation += 1

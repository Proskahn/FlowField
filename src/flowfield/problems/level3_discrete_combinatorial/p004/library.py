"""Exact 7x7 Hamiltonian library from Zhou et al. Algorithm 1; stable zero-based IDs."""
from functools import lru_cache
import gzip
import hashlib
import json
from numbers import Real
from pathlib import Path
import math

GRID=7
COUNT=28348
DOI='10.1016/j.jpowsour.2024.235753'
LIBRARY_REVISION='hamilton-7x7-fixed-ports-v1'


def validate_path(path,n=7):
    if isinstance(path,(str,bytes)):
        raise ValueError('A path must be a sequence of integer node IDs')
    values=list(path)
    if len(values)!=n*n:raise ValueError(f'Expected {n*n} ordered node IDs')
    if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or int(v)!=v for v in values):
        raise ValueError('Node IDs must be finite integers; no rounding is performed')
    values=[int(v) for v in values]
    if sorted(values)!=list(range(n*n)):raise ValueError('Path must visit every grid node exactly once')
    if values[:2]!=[0,n] or values[-2:]!=[n*n-1-n,n*n-1]:
        raise ValueError('Required first edge is (1,1)->(1,2); last edge is (7,6)->(7,7)')
    if any(abs(a%n-b%n)+abs(a//n-b//n)!=1 for a,b in zip(values,values[1:])):
        raise ValueError('Consecutive nodes must be orthogonal grid neighbors')
    return values


def enumerate_paths(n=7):
    """Pure-Python DFS with endpoint, forced-move and connectivity pruning.

    Sorting after enumeration gives stable IDs independent of traversal order.
    With these opposite-corner endpoints, even n has no Hamiltonian path by parity.
    """
    if type(n) is not int or not 2<=n<=7:raise ValueError('Enumeration supports grid sizes 2 through 7')
    if n%2==0:return []
    end=n*n-1;last=end-n;allmask=(1<<(n*n))-1
    adjacent=[]
    for v in range(n*n):
        x,y=v%n,v//n
        adjacent.append(tuple(yy*n+xx for xx,yy in [(x,y+1),(x+1,y),(x,y-1),(x-1,y)] if 0<=xx<n and 0<=yy<n))
    masks=[sum(1<<v for v in row) for row in adjacent];paths=[];path=[0,n]
    def dfs(v,visited):
        if visited==allmask:
            if v==end:paths.append(tuple(path))
            return
        remaining=allmask^visited;x,y=v%n,v//n
        def blocked(xx,yy):return not(0<=xx<n and 0<=yy<n) or bool(visited&(1<<(yy*n+xx)))
        l,r,u,d=blocked(x-1,y),blocked(x+1,y),blocked(x,y+1),blocked(x,y-1)
        if l and r and not u and not d or u and d and not l and not r:return
        choices=[];forced=[]
        for w in adjacent[v]:
            if not (remaining>>w)&1:continue
            if w==end:
                if remaining==(1<<end) and v==last:choices.append(w)
                continue
            degree=(masks[w]&remaining).bit_count()
            if degree==0:return
            if degree==1:forced.append(w)
            choices.append(w)
        if len(forced)>1:return
        if forced:choices=forced
        if len(path)%8==0:
            reach=1<<v;front=reach;allowed=remaining|reach
            while front:
                nxt=0
                while front:
                    bit=front&-front;front^=bit;nxt|=masks[bit.bit_length()-1]
                front=nxt&allowed&~reach;reach|=front
            if reach!=allowed:return
        for w in choices:
            if w==last and remaining!=((1<<end)|(1<<last)):continue
            path.append(w);dfs(w,visited|(1<<w));path.pop()
    dfs(n,1|(1<<n))
    return sorted(paths)


@lru_cache(maxsize=1)
def catalog():
    folder=Path(__file__).parent/'data';data=(folder/'paths.bin.gz').read_bytes()
    manifest=json.loads((folder/'library.json').read_text())
    if hashlib.sha256(data).hexdigest()!=manifest['compressed_sha256']:raise ValueError('Library checksum mismatch')
    raw=gzip.decompress(data)
    if len(raw)!=COUNT*49 or hashlib.sha256(raw).hexdigest()!=manifest['paths_sha256']:
        raise ValueError('Invalid P004 library')
    paths=tuple(tuple(raw[i:i+49]) for i in range(0,len(raw),49))
    if len(set(paths))!=COUNT or paths!=tuple(sorted(paths)):raise ValueError('Noncanonical library ordering')
    for p in paths:validate_path(p)
    return paths


@lru_cache(maxsize=1)
def indices():return {p:i for i,p in enumerate(catalog())}


def path_from_id(identifier):
    if isinstance(identifier,bool) or not isinstance(identifier,Real) or not math.isfinite(identifier) or int(identifier)!=identifier or not 0<=identifier<COUNT:
        raise ValueError(f'Library ID must be an integer in [0,{COUNT-1}]')
    return list(catalog()[int(identifier)])


def path_id(path):return indices()[tuple(validate_path(path))]


def serpentine():
    return [y*7+x for x in range(7) for y in (range(7) if x%2==0 else range(6,-1,-1))]


def features(path):
    """Eqs. 11–13 in lattice-edge units; P4 ambiguity is exposed, not hidden."""
    p=validate_path(path);directions=[(b%7-a%7,b//7-a//7) for a,b in zip(p,p[1:])]
    runs=[];last=None
    for v in directions:
        if v!=last:runs.append(1);last=v
        else:runs[-1]+=1
    mean=sum(runs)/len(runs);positions={v:i for i,v in enumerate(p)}
    # Each undirected lattice edge once, equivalent to 1/2 of the directed sum.
    distances=[abs(positions[a]-positions[b]) for a in range(49) for b in (a+1,a+7)
               if b<49 and (b==a+7 or a%7<6)]
    return dict(corners=len(runs)-1,straight_lengths_lattice=runs,
        straight_std_lattice=math.sqrt(sum((v-mean)**2 for v in runs)/len(runs)),
        adjacent_length_products_lattice2=sum(a*b for a,b in zip(runs,runs[1:])),
        shortcut_distance_sum_lattice=sum(distances),
        shortcut_savings_sum_lattice=sum(v-1 for v in distances),
        mean_neighbor_path_distance_lattice=sum(distances)/len(distances),
        mean_node_summed_neighbor_distance_lattice=2*sum(distances)/49,
        p4_convention='literal Eq.13 sum over 84 undirected grid edges; Table4 normalization unresolved')


def coordinates(path):return [[v%7+1,v//7+1] for v in validate_path(path)]


def ml_inputs(path):return [c for xy in coordinates(path) for c in xy]

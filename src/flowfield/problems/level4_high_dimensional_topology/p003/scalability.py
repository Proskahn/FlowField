"""Explicit scalable P003 instances on a fixed analysis mesh; no CFD calls."""

from dataclasses import replace
import json
import math
from pathlib import Path

from .config import SCALABLE_MAPPING, TopologyConfig
from .design import grid, stencil


def scalable_config(*, dimension=None, shape=None, mesh='medium', stage=1,
                    filter_radius_m=.00035, iterations=10000):
    """Choose exact rectangular controls, preserving plate, ports and radius.

    ``shape`` is (rows, columns) of points, not intervals. For ``dimension``,
    choose the factorization with the closest physical point spacings among
    those supported by the CFD mesh. No rounding or unused padding variables.
    """
    if (dimension is None) == (shape is None):
        raise ValueError('Supply exactly one of dimension or shape')
    base=TopologyConfig(mesh=mesh, stage=stage, design_mapping=SCALABLE_MAPPING,
                        filter_radius_m=filter_radius_m,iterations=iterations,
                        design_nx=2,design_ny=2)
    x,y=grid(base)
    max_rows,max_cols=len(y)-1,len(x)-1
    if dimension is not None:
        if type(dimension) is not int or dimension < 9:
            raise ValueError('dimension must be an integer >= 9')
        if dimension > max_rows*max_cols:
            raise ValueError(f'{dimension} variables exceed the {max_rows*max_cols} CFD cells')
        candidates=[(dimension//cols,cols) for cols in range(3,max_cols+1)
                    if dimension%cols == 0 and 3 <= dimension//cols <= max_rows]
        if not candidates:
            raise ValueError(f'Cannot represent exactly {dimension} variables by a rectangular grid '
                             f'with 3..{max_rows} rows and 3..{max_cols} columns; '
                             'choose another dimension (for example, 500 or 512 on the medium mesh)')
        rows,cols=min(candidates,key=lambda pair: (
            abs(math.log((base.length_m/(pair[0]-1))/(base.width_m/(pair[1]-1)))),pair))
    else:
        if (not isinstance(shape,(tuple,list)) or len(shape) != 2
                or any(type(v) is not int or v < 3 for v in shape)):
            raise ValueError('shape must contain two integers >= 3: rows, columns')
        rows,cols=shape
    return replace(base,design_nx=cols-1,design_ny=rows-1)


def add_dimension_argument(parser):
    parser.add_argument('--dimension',type=int,metavar='N',
        help='P003 variable count; automatically selects the medium CFD mesh and scalable mapping '
             '(e.g. 500, 512, 1000, 2000, 3000, 4000, 5000, 6000); requires a rectangular factorization')


def config_from_dimension(dimension, *, source=None, mesh=None):
    if source is not None:
        raise ValueError('Use --dimension or --config, not both')
    return scalable_config(dimension=dimension,mesh=mesh or 'medium')


def grid_summary(config):
    """Small run-plan description; no filtering or CFD evaluation required."""
    x,y=grid(config)
    return dict(mesh=config.mesh,design_mapping=config.design_mapping,
                design_points=[config.design_ny+1,config.design_nx+1],
                cfd_shape=[len(y)-1,len(x)-1],cfd_cells=(len(y)-1)*(len(x)-1),
                config_sha256=config.fingerprint)


def mapping_report(config):
    """Inspect support and size; cell counts are not a numerical rank test."""
    x,y=grid(config)
    rows,areas=stencil(config)
    columns=[0]*config.dimension
    for row in rows:
        for node,weight in row:
            if weight > 0:columns[node]+=1
    unsupported=sum(count == 0 for count in columns)
    if unsupported:
        raise ValueError(f'{unsupported} design variables have no influence on the CFD material field')
    dx=config.width_m/config.design_nx
    dy=config.length_m/config.design_ny
    return dict(problem='P003',dimension=config.dimension,
        design_points=[config.design_ny+1,config.design_nx+1],
        design_mapping=config.design_mapping,mesh=config.mesh,
        cfd_shape=[len(y)-1,len(x)-1],cfd_cells=len(areas),
        physical_size_m=[config.length_m,config.width_m],
        port_widths_m=dict(inlet=config.inlet_width_m,outlet=config.outlet_width_m),
        filter_radius_m=config.filter_radius_m,design_spacing_m=dict(x=dx,y=dy),
        solver_iteration_limit=config.iterations,
        spacing_aspect_ratio=max(dx,dy)/min(dx,dy),
        stage=config.stage,p=config.interpolation['simp_p'],q=config.interpolation['ramp_q'],
        mapping_nonzeros=sum(len(row) for row in rows),unused_variables=unsupported,
        minimum_cell_support=min(columns),
        rank_status='not assessed; nonzero support and cell counts do not prove independence',
        cfd_validation='not run',config_sha256=config.fingerprint)


def add_arguments(parser):
    shape=parser.add_mutually_exclusive_group(required=True)
    shape.add_argument('--dimension',type=int,help='Exact number of variables; requires a supported rectangular factorization')
    shape.add_argument('--shape',type=int,nargs=2,metavar=('ROWS','COLS'),help='Explicit number of design points on each axis')
    parser.add_argument('--mesh',choices=('coarse','medium','fine'),default='medium')
    parser.add_argument('--stage',type=int,choices=range(4),default=1,help='Fixed material stage (default 1)')
    parser.add_argument('--filter-radius',type=float,default=.00035,help='Physical filter radius in metres (default 0.00035)')
    parser.add_argument('--iterations',type=int,default=10000,help='Maximum CFD solver iterations; convergence tolerances remain unchanged')
    parser.add_argument('--output',type=Path,required=True,help='New configuration JSON; existing files are never overwritten')
    parser.add_argument('--dry-run',action='store_true',help='Inspect the mapping without writing files or running CFD')


def execute(args):
    config=scalable_config(dimension=args.dimension,shape=args.shape,mesh=args.mesh,
                           stage=args.stage,filter_radius_m=args.filter_radius,iterations=args.iterations)
    report=mapping_report(config)
    if not args.dry_run:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(config.to_dict(),stream,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(report,config_file=str(args.output.resolve()),
                          status='planned' if args.dry_run else 'configuration_written'),indent=2))
    return 0

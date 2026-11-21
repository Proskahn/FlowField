"""Layered 3D mesh; voxel resolution of Table 2 profiles is recorded explicitly."""
import math
from collections import Counter
from .config import FLOW_FIELDS
from ..p008.geometry import header,write,field
ZONES=('CBP','CCH','CGDL','CCL','PEM','ACL','AGDL','ACH','ABP')
FLUID={1,2,3,5,6,7}
PATCHES=('anodeInlet','anodeOutlet','cathodeInlet','cathodeOutlet','anodeTerminal','cathodeTerminal','externalWalls')

def esc_width(x,variant):
    # S2, S3 and unit count from Table 2/Fig. 2. A symmetric 0.5-mm
    # end allowance closes the unreported ESCFF-1 straight transition.
    s2,s3,n={'ESCFF-1':(4.25,0.,1),'ESCFF-2':(2.,.5,2),
             'ESCFF-3':(1.25,.5,3),'ESCFF-4':(.875,.5,4)}[variant]
    u=abs(x-11.5)-.5
    if u<=0 or u>=9.5:return 1.
    u-=max(0.,(9.5-(n*2*s2+(n+1)*s3))/2)+s3
    if u<=0:return 1.
    period=2*s2+s3;index=int(u/period)
    if index>=n:return 1.
    phase=u-index*period
    return 1.-.5*phase/s2 if phase<s2 else .5+.5*(phase-s2)/s2 if phase<2*s2 else 1.

def channel(x,y,variant):
    if variant.startswith('ESCFF'):
        rows=[1.+1.5*i for i in range(15)];width=esc_width(x,variant)
        if .5<=x<=22.5 and any(abs(y-r)<=width/2+1e-10 for r in rows):return True
        if 11<=x<=12 and (y<=1 or y>=22):return True  # Two perimeter outlets.
        for i in range(14):
            # Four mirrored serpentine paths leave the middle feed and merge
            # at the top/bottom central outlet, as drawn in Figs. 1-3.
            outer=(i%2==0 if i<7 else i%2==1)
            turn=(.5<=x<=1.5 or 21.5<=x<=22.5) if outer else 11<=x<=12
            if turn and rows[i]<=y<=rows[i+1]:return True
        return False
    if variant=='serpentine':
        columns=[1+2.1*i for i in range(11)]
        if .5<=y<=22.5 and any(abs(x-v)<=.5+1e-10 for v in columns):return True
        for i in range(10):
            if columns[i]<=x<=columns[i+1] and (21.5<=y<=22.5 if i%2==0 else .5<=y<=1.5):return True
        return (y<=1 and abs(x-columns[0])<=.5) or (y>=22 and abs(x-columns[-1])<=.5)
    # Reference CDLIS dimensions are not tabulated; declared reconstruction:
    # twin mirrored 20-pass serpentine channels, triangular width .35--.75 mm.
    u=abs(x-11.5);r=(u%2.5)/2.5;width=.35+.4*abs(2*r-1)
    rows=[1.+21*i/19 for i in range(20)]
    if .5<=x<=22.5 and u>=.5 and any(abs(y-v)<=width/2 for v in rows):return True
    for i in range(19):
        turn=(.5<=x<=1.5 or 21.5<=x<=22.5) if i%2 else (.5<=u<=1.5)
        if turn and rows[i]<=y<=rows[i+1]:return True
    return (y<=1. or y>=22.) and (.5<=x<=1.5 or 21.5<=x<=22.5)

def build_geometry(g,config):
    nx,ny,counts={'coarse':(46,92,(1,2,3,1,2,1,3,2,1)),
        'medium':(69,138,(2,3,4,1,2,1,4,3,2)),
        'fine':(92,184,(2,4,5,2,4,2,5,4,2))}[config.mesh]
    x=[23*i/nx for i in range(nx+1)];y=[23*j/ny for j in range(ny+1)]
    widths=(.5,1.,.3,.0129,.108,.0129,.3,1.,.5)
    z=[0.];layers=[]
    for layer,(width,count) in enumerate(zip(widths,counts)):
        if layer in (2,6):
            weights=[1.5**k for k in range(count)]
            if layer==2:weights.reverse()
            for weight in weights:z.append(z[-1]+width*weight/sum(weights))
        else:
            start=z[-1];z.extend(start+width*k/count for k in range(1,count+1))
        layers.extend([layer]*count)
    mask=[[channel((x[i]+x[i+1])/2,(y[j]+y[j+1])/2,config.flow_field) for i in range(nx)] for j in range(ny)]
    cells=[];ids=[]
    for k,layer in enumerate(layers):
        for j in range(ny):
            for i in range(nx):
                xx,yy=(x[i]+x[i+1])/2,(y[j]+y[j+1])/2
                bore=config.flow_field.startswith('ESCFF') and abs(xx-11.5)<.5 and abs(yy-11.5)<.5
                material=layer
                if layer==1 and not mask[j][i]:material=0
                elif layer==7 and not mask[j][i]:material=8
                elif layer==0 and bore:material=1
                elif layer==8 and bore:material=7
                cells.append((i,j,k));ids.append(material)
    return dict(x=x,y=y,z=z,cells=cells,ids=ids,flow_field=config.flow_field,area=.023**2)

def boundary_patch(m,c,side):
    mat=m['ids'][c];variant=m['flow_field'];i,j,k=m['cells'][c]
    if mat in (1,7):
        electrode='cathode' if mat==1 else 'anode'
        if variant.startswith('ESCFF'):
            if side in (4,5):return electrode+'Inlet'
            if side in (2,3):return electrode+'Outlet'
        elif side==2:return electrode+'Inlet'
        elif side==3:return electrode+'Outlet'
    return 'cathodeTerminal' if side==4 else 'anodeTerminal' if side==5 else 'externalWalls'

def mesh(case, m):
    x, y, z = (m[v] for v in ('x', 'y', 'z'))
    nx, ny, nz = (len(v)-1 for v in (x, y, z))
    points, point_ids, internal = [], {}, []
    boundary = {p: [] for p in PATCHES}
    directions = [((-1,0,0), ((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
                  ((1,0,0), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
                  ((0,-1,0), ((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
                  ((0,1,0), ((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
                  ((0,0,-1), ((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
                  ((0,0,1), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)))]
    inlet_areas = Counter()
    inlet_face_areas = {p: [] for p in PATCHES if p.endswith('Inlet')}
    for c, (i, j, k) in enumerate(m['cells']):
        for side, ((di,dj,dk), corners) in enumerate(directions):
            ii, jj, kk = i+di, j+dj, k+dk
            nbr = (kk*ny+jj)*nx+ii if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz else -1
            if 0 <= nbr < c:
                continue
            face = []
            for a, b, d in corners:
                key = i+a, j+b, k+d
                if key not in point_ids:
                    point_ids[key] = len(points)
                    points.append(key)
                face.append(point_ids[key])
            if nbr >= 0:
                internal.append((face, c, nbr))
            else:
                p = boundary_patch(m, c, side)
                boundary[p].append((face, c, -1))
                if 'Inlet' in p:
                    widths=(x[i+1]-x[i],y[j+1]-y[j],z[k+1]-z[k])
                    axis=side//2
                    area=math.prod(widths[d] for d in range(3) if d!=axis)*1e-6
                    inlet_areas[p] += area
                    inlet_face_areas[p].append(area)
    poly = case/'constant/polyMesh'
    poly.mkdir(parents=True, exist_ok=True)
    write(poly/'points', str(len(points))+'\n(\n'+''.join(
        f'({x[i]*1e-3:.12g} {y[j]*1e-3:.12g} {z[k]*1e-3:.12g})\n' for i,j,k in points)+')\n', 'vectorField')
    all_faces = internal+[f for group in boundary.values() for f in group]
    write(poly/'faces', str(len(all_faces))+'\n(\n'+''.join('4('+ ' '.join(map(str,f))+')\n' for f,_,_ in all_faces)+')\n', 'faceList')
    write(poly/'owner', str(len(all_faces))+'\n(\n'+''.join(f'{o}\n' for _,o,_ in all_faces)+')\n', 'labelList')
    write(poly/'neighbour', str(len(internal))+'\n(\n'+''.join(f'{n}\n' for _,_,n in internal)+')\n', 'labelList')
    offset, btext = len(internal), str(len(PATCHES))+'\n(\n'
    for p, faces in boundary.items():
        typ = 'patch' if ('Inlet' in p or 'Outlet' in p) else 'wall'
        btext += f'{p}\n{{ type {typ}; nFaces {len(faces)}; startFace {offset}; }}\n'
        offset += len(faces)
    write(poly/'boundary', btext+')\n', 'polyBoundaryMesh')
    zonetext = '9\n(\n'
    for n, name in enumerate(ZONES):
        labels = [c for c, zone in enumerate(m['ids']) if zone == n]
        zonetext += f'{name}\n{{ type cellZone; cellLabels List<label> {len(labels)}\n(\n'+ '\n'.join(map(str,labels))+'\n); }\n'
    write(poly/'cellZones', zonetext+')\n', 'regIOobject')
    write(poly/'faceZones', '0\n(\n)\n', 'regIOobject')
    m['inlet_areas'] = dict(inlet_areas)
    m['inlet_face_areas'] = inlet_face_areas
    from ....mesh import PolyMesh
    return PolyMesh.from_case(case)


"""1 mm channels/ribs on a 14x14 mm footprint; Fig.2(c) geometry in SI."""
from .library import validate_path,path_id,coordinates,features,ml_inputs


def decode(path):
    p=validate_path(path)
    points=[(.001+.002*(v%7),.001+.002*(v//7)) for v in p]
    return dict(path=p,library_id=path_id(p),coordinates_1based=coordinates(p),ml_inputs=ml_inputs(p),
        centerline_m=[[.001,0.]]+[list(q) for q in points]+[[.013,.014]],
        channel_width_m=.001,channel_depth_m=.001,rib_width_m=.001,footprint_m=[.014,.014],
        perimeter_margin_m=.0005,channel_area_m2=98e-6,channel_volume_m3=98e-9,
        centerline_length_m=.098,features=features(p),geometry_status='main-paper-dimensions; half-rib perimeter margin and port extensions explicit')


def footprint_cells(path,resolution):
    """Exact unions of node squares, selected edge rectangles and two port stubs."""
    p=validate_path(path);q=resolution
    if type(q) is not int or q<2 or q%2:raise ValueError('Use an even number of cells per channel width')
    cells=set()
    def rectangle(x0,x1,y0,y1):
        for j in range(y0,y1):
            for i in range(x0,x1):cells.add((i,j))
    for v in p:
        cx=q+2*q*(v%7);cy=q+2*q*(v//7)
        rectangle(cx-q//2,cx+q//2,cy-q//2,cy+q//2)
    for a,b in zip(p,p[1:]):
        ax,ay=q+2*q*(a%7),q+2*q*(a//7);bx,by=q+2*q*(b%7),q+2*q*(b//7)
        rectangle(min(ax,bx)-q//2,max(ax,bx)+q//2,min(ay,by)-q//2,max(ay,by)+q//2)
    rectangle(q//2,3*q//2,0,q)
    rectangle(25*q//2,27*q//2,13*q,14*q)
    if len(cells)!=98*q*q:raise ValueError('Channel footprint area is inconsistent')
    return sorted(cells,key=lambda v:(v[1],v[0]))


def svg(geometry):
    """Vector preview of the physical channel, with square turns and flat port ends."""
    points=' '.join(f'{x*1000:g},{14-y*1000:g}' for x,y in geometry['centerline_m'])
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="560" height="560">'
        '<rect width="14" height="14" fill="#50586b"/>'
        f'<polyline points="{points}" fill="none" stroke="#eaf4f7" stroke-width="1" stroke-linejoin="miter" stroke-linecap="butt"/>'
        '</svg>\n')

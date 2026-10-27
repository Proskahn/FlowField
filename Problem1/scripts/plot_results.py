#!/usr/bin/env python3
"""Optional static PNG figures; requires Pillow. Core CFD scripts do not."""
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
FONT_CANDIDATES=['/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']


def font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists(): return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def geometry():
    p=json.loads((ROOT/'config/reference.json').read_text())
    im=Image.new('RGB',(1680,1030),'#f6f8fb'); d=ImageDraw.Draw(im)
    ink='#142c45'; muted='#52677b'; blue='#cee7f8'; red='#bd4b52'
    d.text((65,40),'Anode flow-field VOF reconstruction',font=font(40),fill=ink)
    d.text((65,100),'Li et al. (2025), Fig. 3 topology  |  Plan view  |  Dimensions in mm',font=font(23),fill=muted)
    scale=51
    for col,w in enumerate([2,1,0.5]):
        left=55+col*550
        d.rounded_rectangle((left,162,left+520,877),radius=16,fill='white',outline='#dce4ec',width=2)
        d.text((left+25,184),f'{w:g} mm channel / rib',font=font(29),fill=ink)
        n=round(4/(2*w))
        d.text((left+25,227),f'{n} channel'+('s' if n>1 else '')+'  |  1 mm height',font=font(22),fill=muted)
        x0=left+260-(4-w)/2*scale; y0=770
        def pt(x,y): return (x0+x*scale,y0-y*scale)
        def box(xa,ya,xb,yb,fill,outline=None,width=1):
            a=pt(xa,yb); b=pt(xb,ya);d.rectangle((*a,*b),fill=fill,outline=outline,width=width)
        box(0,8,4,9,blue,ink,2);box(-w,-1,4-w,0,blue,ink,2)
        for c in range(n):
            xa=2*c*w
            box(xa,0,xa+w,8,blue,ink,2)
            for ix in range(round(w/p['pore_pitch_mm'])):
                for iy in range(32):
                    cx=xa+(ix+.5)*.25;cy=(iy+.5)*.25
                    box(cx-.05,cy-.05,cx+.05,cy+.05,red)
            d.text(pt(xa+w/2,4),f'Ch{c+1}',font=font(20),fill=ink,anchor='mm',stroke_width=2,stroke_fill=blue)
        # Draw over common internal edges to make manifold/channel connections clear.
        for c in range(n):
            for yy in (0,8):
                d.line((pt(2*c*w+.025,yy),pt((2*c+1)*w-.025,yy)),fill=blue,width=4)
        def arrow(xa,ya,xb,yb):
            a=pt(xa,ya); b=pt(xb,yb);d.line((a,b),fill=ink,width=4)
            d.polygon([(b[0],b[1]),(b[0]+11,b[1]-7),(b[0]+11,b[1]+7)],fill=ink)
        arrow(3.7,8.5,2.8,8.5);arrow(-w+.9,-.5,-w+.2,-.5)
        d.text(pt(4.12,8.5),'In',font=font(21),fill=ink,anchor='lm')
        d.text(pt(-w-.15,-.5),'Out',font=font(21),fill=ink,anchor='rm')
        a,b=pt(0,9.35),pt(4,9.35);d.line((a,b),fill=muted,width=2)
        for q in [a,b]:d.line((q[0],q[1]-6,q[0],q[1]+6),fill=muted,width=2)
        d.text(pt(2,9.55),'4',font=font(22),fill=muted,anchor='mb')
        a,b=pt(4.4,0),pt(4.4,8);d.line((a,b),fill=muted,width=2)
        for q in [a,b]:d.line((q[0]-6,q[1],q[0]+6,q[1]),fill=muted,width=2)
        d.text(pt(4.58,4),'8',font=font(22),fill=muted,anchor='lm')
        d.text((left+25,842),'256 oxygen openings below channels',font=font(20),fill=muted)
    d.rectangle((65,918,84,937),fill=red)
    d.text((98,914),'Estimated pores: 0.1 mm square, 0.25 mm pitch, 0.1 mm passage length.',font=font(23),fill=ink)
    d.text((65,960),'No oxygen openings beneath manifolds. Pore layout is an assumption; this is not the authors\' CAD.',font=font(22),fill=muted)
    im.save(ROOT/'docs/geometry.png')


def startup():
    series=[]
    for w,color in [('2','#275fc4'),('1','#029181'),('0.5','#bd4b52')]:
        path=ROOT/f'runs/check/w{w}mm/postProcessing/bubbleMetrics/metrics.csv'
        if not path.exists(): return
        with path.open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        series.append((w,color,rows))
    im=Image.new('RGB',(1680,720),'white');d=ImageDraw.Draw(im)
    d.text((65,30),'Computed startup diagnostics',font=font(38),fill='#142c45')
    d.text((65,85),'Coarse mesh, 53,408 cells per design  |  First 1 ms only  |  Not a quasi-steady paper comparison',font=font(23),fill='#52677b')
    panels=[('gas_fraction_channels','Channel gas volume fraction',0.0045),
            ('beta_bottom_solid','Solid-bottom gas coverage',0.006),
            ('oxygen_balance_residual_m3_s','Oxygen balance residual / total feed',0.00002)]
    for j,(key,title,ymax) in enumerate(panels):
        x=100+550*j;y=190;pw=440;ph=375
        d.text((x,y-42),title,font=font(21),fill='#142c45')
        for tick in range(5):
            yy=y+ph-tick*ph/4
            d.line((x,yy,x+pw,yy),fill='#e3e9f0',width=1)
            value=ymax*tick/4
            lab=f'{value:.1e}' if j==2 else f'{value:.4f}'
            d.text((x-10,yy),lab,font=font(17),fill='#52677b',anchor='rm')
        for tick in range(6):
            xx=x+tick*pw/5;d.text((xx,y+ph+13),f'{tick/5:.1f}',font=font(18),fill='#52677b',anchor='mt')
        d.line((x,y,x,y+ph,x+pw,y+ph),fill='#52677b',width=2)
        for w,color,rows in series:
            pts=[]
            for r in rows:
                if r['time_s']>0.001+1e-10:continue
                val=r[key]
                if j==2:val=abs(val)/5.716085132943106e-8
                pts.append((x+pw*r['time_s']/.001,y+ph-ph*val/ymax))
            d.line(pts,fill=color,width=3)
        d.text((x+pw/2,y+ph+47),'Time (ms)',font=font(21),fill='#52677b',anchor='mt')
    for i,(w,color,_) in enumerate(series):
        x=590+i*190;d.line((x,668,x+32,668),fill=color,width=5)
        d.text((x+43,668),f'{w} mm',font=font(23),fill='#142c45',anchor='lm')
    im.save(ROOT/'results/startup_diagnostics.png')


if __name__=='__main__':
    geometry()
    startup()

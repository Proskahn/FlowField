#!/usr/bin/env python3
"""Time-weighted VOF metrics. A short run never passes as quasi-steady validation."""
import argparse
import csv
import json
import math
from pathlib import Path

METRICS = ("beta_bottom_solid", "beta_bottom_with_mouths", "beta_bottom_threshold",
           "beta_top", "beta_side", "gas_fraction_channels", "pressure_drop_Pa")


def interpolate(rows, t, key):
    for a,b in zip(rows,rows[1:]):
        if a['time_s'] <= t <= b['time_s']:
            f=(t-a['time_s'])/(b['time_s']-a['time_s'])
            return a[key]+f*(b[key]-a[key])
    raise ValueError(f"Cannot interpolate t={t}")


def weighted_mean(rows, lo, hi, key):
    if hi <= lo:
        raise ValueError("Need two distinct time samples")
    points=[(lo,interpolate(rows,lo,key))]
    points.extend((r['time_s'],r[key]) for r in rows if lo < r['time_s'] < hi)
    points.append((hi,interpolate(rows,hi,key)))
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points,points[1:]))/(hi-lo)


def read_rows(case):
    with (case/'postProcessing/bubbleMetrics/metrics.csv').open() as f:
        raw=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    if not raw or any(not math.isfinite(v) for r in raw for v in r.values()):
        raise ValueError(f"Empty or nonfinite metrics: {case}")
    # Keep the latest record if a resumed solver re-emits an existing time.
    return sorted({r['time_s']:r for r in raw}.values(),key=lambda r:r['time_s'])


def analyze(case, start=0.2, end=1.0, allow_short=False):
    meta=json.loads((case/'case_metadata.json').read_text())
    rows=read_rows(case)
    full=rows[0]['time_s'] <= start+1e-10 and rows[-1]['time_s'] >= end-1e-10
    if not full and not allow_short:
        raise ValueError(f"{case}: only {rows[0]['time_s']:g}–{rows[-1]['time_s']:g} s available; "
                         f"requested {start:g}–{end:g} s. Use --allow-short for a smoke-test summary.")
    lo,hi=max(start,rows[0]['time_s']),min(end,rows[-1]['time_s'])
    if hi<=lo and allow_short:
        lo,hi=rows[0]['time_s'],rows[-1]['time_s']
    means={k:weighted_mean(rows,lo,hi,k) for k in METRICS}
    p,d=meta['parameters'],meta['derived']
    beta=means['beta_bottom_with_mouths']
    eta=(8.31446261815324*p['temperature_K']/(p['electron_transfer_number']*p['transfer_coefficient']*96485.33212)
         *(-p['bubble_omega1']*math.log1p(-min(beta,1-1e-12))))
    feed=d['oxygen_flow_m3_s']+(1-p['water_inlet_alpha'])*d['inlet_mixture_flow_m3_s']
    amin=min(r['alpha_min'] for r in rows)
    amax=max(r['alpha_max'] for r in rows)
    residual=max(abs(r['oxygen_balance_residual_m3_s']) for r in rows)
    return dict(case=str(case),width_mm=meta['mesh']['channel_width_mm'],cells=meta['mesh']['cells'],
                interval_s=[lo,hi], samples=len(rows), requested_interval_complete=full,
                status=("Requested sampling window complete; stationarity and grid independence still require assessment"
                        if full else "Short transient verification only; not a paper comparison"),
                time_weighted_means=means,
                coupling_from_mean_beta=dict(eta_bubble_V=eta,active_area_factor=(1-beta)**p['bubble_omega2'],
                    note="Evaluated from mean beta; not an electrochemical solve or mean instantaneous overpotential"),
                diagnostics=dict(alpha_min=amin,alpha_max=amax,alpha_bound_tolerance=1e-7,
                                 bounded_within_tolerance=(amin>=-1e-7 and amax<=1+1e-7),
                                 max_net_volume_flux_m3_s=max(abs(r['net_volume_flux_m3_s']) for r in rows),
                                 max_oxygen_balance_residual_m3_s=residual,
                                 max_oxygen_balance_relative_to_feed=residual/feed if feed else None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases',nargs='+',type=Path)
    parser.add_argument('--start',type=float,default=0.2)
    parser.add_argument('--end',type=float,default=1.0)
    parser.add_argument('--allow-short',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('results/summary.json'))
    args=parser.parse_args()
    if args.end <= args.start: parser.error('--end must exceed --start')
    summaries=[analyze(c,args.start,args.end,args.allow_short) for c in args.cases]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summaries,indent=2)+'\n')
    for s in summaries:
        print(f"{s['width_mm']:g} mm: {s['status']}; time={s['interval_s']}; "
              f"alpha=[{s['diagnostics']['alpha_min']:.4g},{s['diagnostics']['alpha_max']:.4g}]")
    print(args.output)


if __name__=='__main__': main()

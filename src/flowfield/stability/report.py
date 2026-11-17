"""Portable JSON, CSV and self-contained HTML reports, without plot dependencies."""
import csv
from html import escape
import io
import json
import math
import os
from pathlib import Path


def atomic(path,text):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(text,encoding='utf-8');os.replace(tmp,path)


def sparkline(audit):
    rows=audit.get('history',{}).get('recent',[]);limits=audit.get('limits',{})
    names=[k for k in ('field_change','local_liquid_balance','local_gas_balance','energy_balance',
                       'mass_error','gas_error','volume_error','oxygen_error','water_error',
                       'max_gas_balance_relative','max_continuity_relative') if k in limits and rows and k in rows[-1]]
    if len(rows)<2 or not names:return residual_plot(audit)
    colours=['#1673b1','#d55e00','#00856a','#a3479a','#bc8500','#4955a4']
    all_y=[math.log10(max(abs(r[k])/limits[k],1e-15)) for r in rows for k in names if k in r]
    lo=min(-1,min(all_y));hi=max(1,max(all_y));w,h=680,150
    def y(v):return h-10-(v-lo)/(hi-lo)*(h-20)
    lines=[f'<line x1="0" x2="{w}" y1="{y(0):.2f}" y2="{y(0):.2f}" stroke="black" stroke-dasharray="4 3"/>']
    legends=[]
    for j,key in enumerate(names):
        colour=colours[j%len(colours)];points=' '.join(f'{i*w/(len(rows)-1):.2f},{y(math.log10(max(abs(r[key])/limits[key],1e-15))):.2f}' for i,r in enumerate(rows))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.5"/>')
        legends.append(f'<span style="color:{colour}">{escape(key)}</span>')
    return '<p>Recent diagnostics / final limits (log scale); dashed line = limit. Initialization is not qualification.</p><svg role="img" aria-label="Recent normalized diagnostics" viewBox="0 0 680 150">'+''.join(lines)+'</svg><p>'+ ' · '.join(legends)+'</p>'


def residual_plot(audit):
    series={k:v['recent'] for k,v in audit.get('log',{}).get('residuals',{}).items() if len(v['recent'])>1}
    if not series:return ''
    lo=min(-1,min(math.log10(max(v,1e-15)) for rows in series.values() for _,v in rows));hi=max(0,max(math.log10(max(v,1e-15)) for rows in series.values() for _,v in rows))
    colours=['#1673b1','#d55e00','#00856a','#a3479a','#bc8500','#4955a4'];lines=[];labels=[]
    for j,(key,rows) in enumerate(series.items()):
        colour=colours[j%len(colours)]
        points=' '.join(f'{i*680/(len(rows)-1):.2f},{140-(math.log10(max(v,1e-15))-lo)/(hi-lo)*130:.2f}' for i,(_,v) in enumerate(rows))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.5"/>')
        labels.append(f'<span style="color:{colour}">{escape(key)}</span>')
    return '<p>Recent linear initial residuals (log scale, each field’s last 100 solves). Linear residuals alone do not qualify the model.</p><svg role="img" aria-label="Recent linear residuals" viewBox="0 0 680 150">'+''.join(lines)+'</svg><p>'+' · '.join(labels)+'</p>'


def write_reports(root,report):
    root=Path(root)
    atomic(root/'report.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    rows=report.get('cases',[]);buf=io.StringIO();writer=csv.writer(buf)
    writer.writerow(['problem','experiment','status','qualified','last_step','stage','case_directory'])
    for r in rows:
        a=r.get('audit',r);writer.writerow([r.get('problem',a.get('problem')),r.get('experiment','inspect'),a['status'],a.get('qualified',False),a.get('last_step',''),a.get('stage',''),a.get('case_directory','')])
    atomic(root/'summary.csv',buf.getvalue())
    pieces=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>FlowField numerical checks</title>',
        '<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#1b2632}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border-bottom:1px solid #ddd;text-align:left;padding:8px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:12px}section{border-top:2px solid #ddd;margin-top:28px}.passed{color:#08794c}.failed,.blocked,.incompatible{color:#b42318}.incomplete,.initializing,.inconclusive{color:#986600}svg{width:100%;max-height:170px}li{margin:5px 0}</style>',
        '<h1>FlowField numerical checks</h1>',f'<p>Overall: <strong>{escape(report.get("status","unknown"))}</strong></p>',
        '<p>A pass applies only to the tested configuration and observation horizon. It does not prove mathematical stability, mesh independence, or agreement with the paper. No optimization is performed.</p>',
        '<table><tr><th>Problem / experiment</th><th>Assessment</th><th>Step / stage</th></tr>']
    for r in rows:
        a=r.get('audit',r);status=a['status'];label=f'{r.get("problem",a.get("problem","?"))} / {r.get("experiment","inspect")}'
        pieces.append(f'<tr><td>{escape(label)}</td><td class="{escape(status)}">{escape(status)}</td><td>{escape(str(a.get("last_step","")))} {escape(a.get("stage",""))}</td></tr>')
    pieces.append('</table>')
    for r in rows:
        a=r.get('audit',r);pieces.append(f'<section><h2>{escape(r.get("problem",a.get("problem","?")))} — {escape(r.get("experiment","inspect"))}</h2>')
        pieces.append(f'<p>{escape(a.get("case_directory", ""))}</p>')
        if a.get('error') or a.get('reason'):pieces.append('<pre>'+escape(str(a.get('error',a.get('reason'))))+'</pre>')
        pieces.append('<table><tr><th>Check</th><th>Status</th><th>Value / limit / evidence</th></tr>')
        for c in a.get('checks',[]):
            value=f'{c.get("value","")} / {c.get("limit","")} ' if 'limit' in c else ''
            pieces.append(f'<tr><td>{escape(c["name"])}</td><td class="{escape(c["status"])}">{escape(c["status"])}</td><td>{escape(value+c["message"])}</td></tr>')
        pieces.append('</table>'+sparkline(a))
        if a.get('warnings'):pieces.append('<ul>'+''.join('<li>'+escape(x)+'</li>' for x in a['warnings'])+'</ul>')
        pieces.append('<details><summary>Full evidence</summary><pre>'+escape(json.dumps(a,indent=2,allow_nan=False))+'</pre></details></section>')
    pieces.append('<h2>Requested comparisons</h2><pre>'+escape(json.dumps(report.get('comparisons',[]),indent=2,allow_nan=False))+'</pre>')
    pieces.append('<h2>Coverage and limits</h2><pre>'+escape(json.dumps(report.get('coverage',{}),indent=2))+'</pre></html>')
    atomic(root/'report.html',''.join(pieces))


def overall(cases,comparisons=()):
    states=[r.get('audit',r)['status'] for r in cases]+[c['status'] for c in comparisons]
    if any(s in ('failed','blocked','incompatible','error') for s in states):return 'failed'
    if any(s not in ('passed','not_applicable') for s in states):return 'inconclusive'
    return 'passed' if 'passed' in states else 'not_applicable'

from pathlib import Path
import csv,json,statistics,hashlib
R=Path(__file__).resolve().parent
names=['cores5_base128','cores20_base64','cores20_base128','cores20_base128x256']
metrics=['Task Duration(us)','aicore_time(us)','aic_mac_time(us)','aic_mte2_time(us)']
configs=[]
for name in names:
 candidates=[]
 for p in sorted((R/'profiles'/name).rglob('op_summary*.csv')):
  with p.open() as f:rows=[r for r in csv.DictReader(f) if r['Op Name']=='matmul_custom']
  if len(rows)==20:candidates.append((p,rows))
 assert candidates,name
 p,rows=candidates[-1]
 rows.sort(key=lambda r:float(r['Task Start Time(us)']))
 used=rows[5:]
 v=json.loads((R/'results'/name/'verification.json').read_text());assert v['passed']
 t=json.loads((R/'results'/name/'tiling.json').read_text())
 assert all(int(r['Block Dim'])==t['blockDim'] and int(r['Mix Block Dim'])==t['usedCoreNum'] for r in rows)
 stats={m:{'median':statistics.median(a:=[float(r[m]) for r in used]),'min':min(a),'max':max(a),'mean':statistics.mean(a),'stdev':statistics.stdev(a),'samples':a} for m in metrics}
 configs.append(dict(name=name,tiling=t,source_csv=str(p.relative_to(R)),source_csv_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),total_calls=20,warmup_excluded=5,samples=15,metrics=stats,verification=v))
s={'actual_date':'2026-11-19','M':4096,'K':4096,'N':5120,'input_dtype':'float16','output_dtype':'float32','input_range':[1,10],'input_metadata':json.loads((R/'data/metadata.json').read_text()),'configurations':configs}
s['speedup_20_vs_5']=configs[0]['metrics']['Task Duration(us)']['median']/configs[2]['metrics']['Task Duration(us)']['median']
s['aicore_speedup_20_vs_5']=configs[0]['metrics']['aicore_time(us)']['median']/configs[2]['metrics']['aicore_time(us)']['median']
(R/'performance_summary.json').write_text(json.dumps(s,indent=2)+'\n')
with (R/'performance_summary.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['config','blockDim','usedCoreNum','baseM','baseN','baseK','samples',*metrics,'max_rel_error','passed'])
 for c in configs:
  t=c['tiling'];w.writerow([c['name'],*[t[k] for k in ('blockDim','usedCoreNum','baseM','baseN','baseK')],15,*[c['metrics'][m]['median'] for m in metrics],c['verification']['max_rel_error'],c['verification']['passed']])
print('WEEK14 | M=4096 K=4096 N=5120 | fp16 inputs -> fp32 output')
print('Ascend910B4 | 20 Cube / 40 Vector | msprof PipeUtilization')
print('5 warmups excluded, 15 measured launches per configuration')
for c in configs:
 t=c['tiling'];print(f"{c['name']}: blocks={t['blockDim']} AIV={t['usedCoreNum']} base=({t['baseM']},{t['baseN']},{t['baseK']})")
 print('  median us: '+', '.join(m+'='+str(c['metrics'][m]['median']) for m in metrics))
 print(f"  correctness: {c['verification']['numel']} elements, failed={c['verification']['failed_elements']}, max_rel_error={c['verification']['max_rel_error']:.9g}")
print(f"20-vs-5 speedup: Task Duration={s['speedup_20_vs_5']:.4f}x, AI Core={s['aicore_speedup_20_vs_5']:.4f}x")
print('ALL FOUR CONFIGURATIONS: PASS')

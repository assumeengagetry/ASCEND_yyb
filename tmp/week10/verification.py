from pathlib import Path
import re,json,hashlib
root=Path(__file__).resolve().parents[2]/'labs/week10_helloworld'
logs=[root/'evidence/03_npu_run.log',root/'evidence/04_output_msg.txt']
results=[]
for path in logs:
 s=path.read_text()
 ids=list(map(int,re.findall(r'DumpHead: AIV-(\d+)',s)))
 assert sorted(ids)==list(range(8)),(path,ids)
 assert s.splitlines().count('Hello World!!!')==8
 assert re.findall(r'block dim=(\d+)',s)==['8']*8
 assert re.findall(r'total_block_num=(\d+)',s)==['8']*8
 assert re.findall(r'CANN Version: ([^,]+)',s)==['8.5.2']*8
 results.append({'log':path.name,'hello_lines':8,'aiv_blocks':ids,'pass':True})
assert 'PROGRAM_EXIT_CODE=0' in logs[0].read_text()
assert '[100%] Built target main' in (root/'evidence/02_build_and_run.log').read_text()
assert '[PASS] Hello World lines: 8 / 8' in (root/'evidence/02_build_and_run.log').read_text()
result={'date':'2026-11-05','soc':'Ascend910B4','CANN':'8.5.2','blockDim':8,'exit_code':0,'runs':results,'passed':True}
(root/'evidence/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))

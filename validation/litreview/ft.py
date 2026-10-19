import json,urllib.request,re,os
m=json.load(open('meta.json'))
os.makedirs('ft',exist_ok=True)
for k,v in m.items():
    p=v.get('pmcid')
    if not p or os.path.exists(f'ft/{k}.txt'): continue
    try:
        x=urllib.request.urlopen(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{p}/fullTextXML",timeout=40).read().decode('utf8','ignore')
        t=re.sub(r'<ref-list.*?</ref-list>','',x,flags=re.S)
        t=re.sub(r'<[^>]+>',' ',t); t=re.sub(r'\s+',' ',t)
        open(f'ft/{k}.txt','w').write(t); print(k,p,len(t))
    except Exception as e: print(k,p,'ERR',e)

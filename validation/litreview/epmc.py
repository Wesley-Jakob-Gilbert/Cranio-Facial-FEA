import json,urllib.request,urllib.parse,sys,re,time
src=json.load(open('../litreview_europepmc.json'))
dois={k:v['doi'] for k,v in src.items()}
extra=json.load(open('extra_dois.json')) if len(sys.argv)>1 else {}
dois.update(extra)
out={}
try: out=json.load(open('meta.json'))
except: pass
for k,d in dois.items():
    if k in out and out[k].get('title'): continue
    q=urllib.parse.quote(f'DOI:"{d}"')
    try:
        js=json.load(urllib.request.urlopen(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&resultType=core&format=json",timeout=30))
        r=js['resultList']['result']
        if not r: out[k]={'doi':d,'err':'noresult'}; continue
        x=r[0]
        out[k]={'doi':d,'title':x.get('title'),'authors':x.get('authorString'),'year':x.get('pubYear'),'journal':(x.get('journalInfo') or {}).get('journal',{}).get('title'),'pmid':x.get('pmid'),'pmcid':x.get('pmcid'),'oa':x.get('isOpenAccess'),'abstract':re.sub('<[^>]+>','',x.get('abstractText','') or '')}
    except Exception as e:
        out[k]={'doi':d,'err':str(e)}
json.dump(out,open('meta.json','w'),indent=1)
for k,v in out.items(): print(k,v.get('year'),v.get('pmid'),v.get('pmcid'),v.get('oa'),len(v.get('abstract') or ''),v.get('err',''),(v.get('authors') or '')[:50])

"""Cache pinned author duplicate-name lists; never execute downloaded code."""
import json, urllib.request, urllib.parse
from research.common import ROOT,sha256,write_json

CACHE=ROOT/'.cache/isic_duplicate_references'
REPO='mmu-dermatology-research/isic_duplicate_removal_strategy'


def fetch(url,limit=3_000_000):
    request=urllib.request.Request(url,headers={'User-Agent':'MinorProject-research-audit'})
    with urllib.request.urlopen(request,timeout=30) as response: data=response.read(limit+1)
    assert len(data)<=limit
    return data


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    manifest=CACHE/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        assert all(sha256(CACHE/r['local_file'])==r['sha256'] for r in saved['files'])
        print(json.dumps(saved,indent=2));return
    commit=json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'))
    assert not tree['truncated']
    files=[]
    paths=[r['path'] for r in tree['tree'] if r['type']=='blob' and
           (r['path'].startswith('file_lists/') and r['path'].endswith('.txt') or r['path'] in ['DISCLAIMER','README.md'])]
    for i,path in enumerate(sorted(paths)):
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/'+urllib.parse.quote(path,safe='/')
        data=fetch(url)
        local=f'{i:02d}_{path.rsplit("/",1)[-1]}'
        (CACHE/local).write_bytes(data)
        files.append(dict(repo_path=path,url=url,local_file=local,sha256=sha256(CACHE/local),bytes=len(data)))
    saved=dict(repository=f'https://github.com/{REPO}',commit=commit,files=files,
        purpose='Published duplicate-name exclusions only. No source code executed, no local files deleted, no test labels loaded.')
    write_json(manifest,saved);print(json.dumps(saved,indent=2))


if __name__=='__main__':main()

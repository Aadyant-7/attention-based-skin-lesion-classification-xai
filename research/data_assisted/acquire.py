"""Download selected external JPEG members only, using guarded HTTP ZIP ranges."""
import argparse,io,json,struct,threading,time,zipfile,zlib
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
from urllib3.exceptions import HTTPError as UrllibHTTPError
import pandas as pd
from PIL import Image
from research.common import ROOT,sha256,relative,write_json,write_csv

OUT=ROOT/'results/data_assisted/clearance_v1'
RAW=ROOT/'data/raw/ISIC2019_external/clearance_v1'
CACHE=ROOT/'.cache/isic2019_zip_index'
URL='https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip'
LOCAL=threading.local()


def session():
    if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
    return LOCAL.session


class RemoteZip(io.RawIOBase):
    def __init__(self):
        response=session().head(URL,timeout=(10,30));response.raise_for_status()
        self.size=int(response.headers['Content-Length']);self.etag=response.headers['ETag']
        assert self.size>1_000_000_000 and response.headers.get('Accept-Ranges')=='bytes'
        self.pos=0;self.transferred=0;self.lock=threading.Lock()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        assert self.pos>=0;return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        data=self.range(self.pos,n);self.pos+=n;return data
    def range(self,start,n):
        assert 0<=start<self.size and 0<n<=32_000_000 and start+n<=self.size
        for attempt in range(3):
            try:
                with session().get(URL,headers={'Range':f'bytes={start}-{start+n-1}','If-Match':self.etag},
                                   stream=True,timeout=(10,45)) as response:
                    # Never silently download the full9GB archive when ranges fail.
                    assert response.status_code==206, f'Range request returned {response.status_code}; refusing full archive'
                    assert response.headers.get('Content-Range','').startswith(f'bytes {start}-{start+n-1}/')
                    data=response.raw.read(n+1)
                assert len(data)==n
                with self.lock:self.transferred+=len(data)
                return data
            except (requests.RequestException,UrllibHTTPError,TimeoutError):
                if attempt==2:raise
        raise RuntimeError('Range request failed')


def main(all_images=False,workers=4):
    assert 1<=workers<=16
    summary=json.loads((OUT/'summary.json').read_text())
    manifest=OUT/'selected_one_view_manifest.csv'
    assert sha256(manifest)==summary['selected_manifest_sha256']
    selected=pd.read_csv(manifest)
    assert selected.image_id.is_unique and selected.lesion_id.is_unique
    assert (selected.source=='BCN').all() and selected.image_id.str.fullmatch(r'ISIC_\d+').all()
    assert not selected.image_id.str.endswith('_downsampled').any()
    RAW.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    remote=RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        lookup={i.filename.rsplit('/',1)[-1].removesuffix('.jpg'):i for i in archive.infolist() if i.filename.endswith('.jpg')}
    assert len(lookup)>=len(selected) and set(selected.image_id)<=set(lookup)
    total=sum(lookup[i].compress_size for i in selected.image_id)
    info=dict(url=URL,etag=remote.etag,archive_bytes=remote.size,selected_images=len(selected),
        selected_compressed_bytes=total,selected_uncompressed_bytes=sum(lookup[i].file_size for i in selected.image_id),
        selected_manifest_sha256=sha256(manifest),range_policy='206 exact byte ranges only; no full archive fallback',
        test_images_requested=False,gpu_used=False)
    existing=OUT/'archive_preflight.json'
    if existing.exists():assert json.loads(existing.read_text())==info
    else:write_json(existing,info)
    assert total<2_000_000_000, 'Selected payload exceeds fixed2GB acquisition budget; no bulk download'
    # Two fixed seed-hash-selected distinct lesion groups per class for transport/format QA.
    sample=selected.sort_values('selection_hash').groupby('class_name',sort=True).head(2)
    frame=selected if all_images else sample
    start=time.perf_counter();records=[]
    print(json.dumps(dict(stage='full_selected_acquisition' if all_images else '14_image_transport_sample',**info)),flush=True)
    def acquire(row):
        assert time.perf_counter()-start<1200,'Hard20-minute acquisition budget; completed files preserved'
        item=lookup[row.image_id];path=RAW/f'{row.image_id}.jpg'
        assert not item.flag_bits&1 and item.file_size<16_000_000
        if path.exists():
            data=path.read_bytes();assert len(data)==item.file_size and zlib.crc32(data)&0xffffffff==item.CRC
        else:
            header=remote.range(item.header_offset,30)
            fields=struct.unpack('<4s5H3I2H',header)
            assert fields[0]==b'PK\x03\x04' and fields[3]==item.compress_type
            name_len,extra_len=fields[-2:]
            block=remote.range(item.header_offset+30,name_len+extra_len+item.compress_size)
            assert block[:name_len].decode('utf-8')==item.filename
            compressed=block[name_len+extra_len:]
            data=zlib.decompress(compressed,-15) if item.compress_type==zipfile.ZIP_DEFLATED else compressed
            assert item.compress_type in [zipfile.ZIP_DEFLATED,zipfile.ZIP_STORED]
            assert len(data)==item.file_size and zlib.crc32(data)&0xffffffff==item.CRC
            temp=path.with_suffix('.jpg.part');temp.write_bytes(data)
            assert not path.exists();temp.replace(path)
        with Image.open(io.BytesIO(data)) as im:
            width,height=im.size;mode=im.mode;im.verify()
        assert 0<width<=10000 and 0<height<=10000 and width*height<60_000_000
        return dict(image_id=row.image_id,lesion_id=row.lesion_id,class_name=row.class_name,
            external_partition=row.external_partition,path=relative(path),sha256=sha256(path),
            archive_crc32=item.CRC,bytes=len(data),width=width,height=height,mode=mode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(acquire,r) for r in frame.itertuples(index=False)]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%100==0 or len(records)==len(frame):
                print(json.dumps(dict(completed=len(records),target=len(frame),seconds=round(time.perf_counter()-start,1),network_bytes=remote.transferred)),flush=True)
    assert set(r['image_id'] for r in records)==set(frame.image_id)
    write_csv(OUT/('acquired_image_manifest.csv' if all_images else 'transport_sample_manifest.csv'),sorted(records,key=lambda r:r['image_id']))
    write_json(OUT/('acquisition_summary.json' if all_images else 'transport_sample_summary.json'),dict(status='completed',images=len(records),
        expected_images=len(frame),runtime_seconds=time.perf_counter()-start,network_bytes=remote.transferred,
        bytes_on_disk=sum(r['bytes'] for r in records),archive_etag=remote.etag,test_images_requested=False,test_loaded=False,
        gpu_used=False,training_launched=False,download_workers=workers,full_pixel_duplicate_gate_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true')
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    main(args.all,args.workers)

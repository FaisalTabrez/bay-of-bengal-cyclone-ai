from pathlib import Path
import json
import fitz
from PIL import Image, ImageOps, ImageDraw

root=Path(__file__).resolve().parent
qa=root/'qa';qa.mkdir(exist_ok=True)
springer=root/'drafts/main.pdf'
if springer.exists():
    (root/'drafts/IEC2026_Springer_Draft.pdf').write_bytes(springer.read_bytes())
pdfs=list(qa.glob('*.pdf'))+[root/'drafts/IEC2026_Springer_Draft.pdf']
result={}
for path in pdfs:
    d=fitz.open(path); folder=qa/path.stem; folder.mkdir(exist_ok=True)
    texts=[]; images=[]; outside=[]
    for i,page in enumerate(d):
        text=page.get_text();texts.append(text)
        png=folder/f'page-{i+1}.png'
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(png)
        im=Image.open(png).convert('RGB'); im.thumbnail((338,478))
        canvas=Image.new('RGB',(358,510),'#dddddd'); canvas.paste(im,((358-im.width)//2,20))
        ImageDraw.Draw(canvas).text((12,492),f'Page {i+1}',fill='black')
        images.append(canvas)
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:continue
            x0,y0,x1,y1=block['bbox']
            if x0 < -1 or y0 < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
                outside.append({'page':i+1,'bbox':[x0,y0,x1,y1]})
    (folder/'text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
    # Contact sheets are inspection aids; full page PNGs remain available.
    for start in range(0,len(images),6):
        batch=images[start:start+6]; sheet=Image.new('RGB',(358*3,510*((len(batch)+2)//3)),'white')
        for j,im in enumerate(batch):sheet.paste(im,((j%3)*358,(j//3)*510))
        sheet.save(folder/f'contact-{start//6+1}.png')
    assert not outside,(path,outside)
    assert 'Results on hold' in '\n'.join(texts) if 'Extended_Abstract' not in path.name else 'on hold' in '\n'.join(texts)
    if 'Springer' in path.name:assert '[?' not in '\n'.join(texts)
    result[path.name]={'pages':len(d),'page_points':[d[0].rect.width,d[0].rect.height],
                       'text_outside_page':outside,'unresolved_citation_marker':'[?' in '\n'.join(texts)}
(qa/'layout_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

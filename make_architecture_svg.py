"""Generate a vector architecture diagram exactly matching RGBDRevisedNet's stage usage."""
from pathlib import Path
import html
ROOT=Path(__file__).parent
items=[]
def rect(x,y,w,h,text,fill,stroke='#42536c',fontsize=17):
    items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
    lines=text.split('|')
    y0=y+h/2-(len(lines)-1)*12
    for i,t in enumerate(lines):
        items.append(f'<text x="{x+w/2}" y="{y0+i*24}" text-anchor="middle" font-size="{fontsize}" font-weight="600" fill="#1a2a44">{html.escape(t)}</text>')
def arrow(x1,y1,x2,y2,dash=False,color='#4b647f',sw=2.7):
    d=f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#arrow)"'+(' stroke-dasharray="7,6"' if dash else '')+'/>'
    items.append(d)
def txt(x,y,t,size=15,color='#58667a',anchor='start'):
    items.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{html.escape(t)}</text>')

# Structure groups
rect(55,157,160,72,'RGB image|3 channels','#edf5ff')
rect(55,564,160,72,'Depth image|1 channel','#eff9f0')
for i,lab in enumerate(['RGB stage 1|128 × 96²','RGB stage 2|256 × 48²','RGB stage 3|512 × 24²','RGB stage 4|1024 × 12²']):
    rect(265+i*205,157,167,72,lab,'#e5effe')
    if i>0:arrow(265+(i-1)*205+167,193,265+i*205,193)
arrow(215,193,265,193)
for i,lab in enumerate(['Depth stem|128 × 96²','Depth layer 2|256 × 48²','Depth layer 3|512 × 24²']):
    rect(265+i*205,564,167,72,lab,'#e8f5e9')
    if i>0:arrow(265+(i-1)*205+167,600,265+i*205,600)
arrow(215,600,265,600)
rect(265,382,167,82,'PG-DUM 1|RGB+Depth edges|μ, logσ², Conf','#f2e9ff','#8b69bc',15)
rect(470,382,167,82,'PG-DUM 2|RGB+Depth edges|μ, logσ², Conf','#f2e9ff','#8b69bc',15)
rect(265,278,167,70,'Shallow guided|injection','#edf5ff')
rect(470,278,167,70,'DAPF stage 2|route 0 / 1 / 2','#fff3dd','#b88323',15)
rect(675,278,167,70,'DAPF stage 3|route 0 / 1 / 2','#fff3dd','#b88323',15)
rect(880,278,167,70,'RGB semantic|bypass','#edf5ff')
for i in range(4):
    arrow(348+i*205,229,348+i*205,278)
for i in range(3):
    x=348+i*205
    if i<2:
        arrow(x,564,x,464,color='#398a52')
        arrow(x,382,x,348,color='#9062c7')
    else:
        # depth3 directly participates in stage3 routing
        arrow(x,564,x,348,color='#398a52')
# Stage 1 and 2 RGB features also enter PG-DUM independently.
# Route at the left edge to avoid incorrectly suggesting feedback cycles.
arrow(268,229,268,382,color='#9062c7',sw=1.9)
arrow(473,229,473,382,color='#9062c7',sw=1.9)
# stage2 confidence -> stage3 DAPF
arrow(637,424,759,348,dash=True,color='#9062c7',sw=2)
txt(654,406,'propagated confidence',12)
# Decoder overview, dashed arrows merge
rect(1120,310,185,110,'Top-down decoder|4 saliency heads|4 boundary heads','#e6f4f7','#45808a',16)
for i in range(4):
    arrow(432+i*205,313,1120,328+i*24,dash=True,color='#4b647f',sw=1.9)
rect(1360,293,177,69,'Saliency map|Ŝ','#ecf7f2','#569c7d')
rect(1360,380,177,69,'Boundary map|B̂','#fff0ed','#ba756f')
arrow(1305,344,1360,328)
arrow(1305,398,1360,414)
txt(55,102,'RGB-centric hierarchical encoder (Swin Transformer)',24,'#1a2a44')
txt(265,531,'Lightweight depth CNN',22,'#1a2a44')
txt(55,697,'Solid: feature input / dependency. Dashed: confidence propagation or decoder skip flow.',16)
txt(55,733,'PG-DUM: Prior-Guided Depth Uncertainty Modeling     DAPF: Difficulty-Aware Progressive Fusion',16)
txt(55,770,'Architecture design prototype. Not a trained/evaluated result. Stage sizes assume 384 × 384 input.',15,'#6b7280')
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="820" viewBox="0 0 1600 820">
<defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 z" fill="#4b647f"/></marker></defs>
<rect width="1600" height="820" fill="#ffffff"/>
<text x="55" y="46" font-size="30" font-weight="700" fill="#1a2a44">RGB-D Structure-Prior Uncertainty and Difficulty-Aware Fusion</text>
'''+'\n'.join(items)+'\n</svg>\n'
(ROOT/'framework_actual_prototype.svg').write_text(svg,encoding='utf-8')
print('Saved SVG',ROOT/'framework_actual_prototype.svg')

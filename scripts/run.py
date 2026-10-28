from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

import warnings

warnings.filterwarnings('ignore')

model_id = 'openai/clip-vit-base-patch32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

paths = sorted(p for p in Path('data/sample').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))

if not paths:
    raise SystemExit('data/sample에 이미지를 넣어주세요.')

images = []
for path in paths:
    with Image.open(path) as img:
        images.append(ImageOps.exif_transpose(img).convert('RGB'))

proc = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device).eval()

query = input('영문 검색어: ').strip()

if not query:
    raise SystemExit('검색어를 입력해주세요.')

inputs = proc(images=images, text=[query], padding=True, truncation=True, return_tensors='pt').to(device)

with torch.inference_mode():
    outputs = model(**inputs)
    img_vecs = F.normalize(outputs.image_embeds.float(), dim=-1)
    txt_vec = F.normalize(outputs.text_embeds.float(), dim=-1)
    scores = (txt_vec @ img_vecs.T)[0]

vals, inds = scores.topk(min(5, len(paths)))
for rank, (score, index) in enumerate(zip(vals.tolist(), inds.tolist()), 1):
    print(rank, paths[index], round(score, 4))

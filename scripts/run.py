import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import torch

from encoder import load_encoder, encode_images, encode_texts
from img_io import load_images
from search import search_vectors

model_id = 'openai/clip-vit-base-patch32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

paths = sorted(p for p in Path('data/sample').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))

if not paths:
    raise SystemExit('data/sample에 이미지를 넣어주세요.')

images = load_images(paths)
proc, model = load_encoder(model_id, device)

img_vecs = encode_images(model, proc, images, device)

while(True):
    query = input('영문 검색어: ').strip()

    if not query:
        raise SystemExit('검색어를 입력해주세요.')
    if query == 'quit':
        break

    txt_vec = encode_texts(model, proc, query, device)

    vals, inds = search_vectors(txt_vec, img_vecs)
    for rank, (score, index) in enumerate(zip(vals.tolist(), inds.tolist()), 1):
        print(rank, paths[index], round(score, 4))
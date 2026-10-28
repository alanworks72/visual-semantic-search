# Visual Semantic Search

자연어나 이미지로 비슷한 이미지를 찾는 검색 프로젝트입니다.
CLIP과 SigLIP2를 활용해 이미지 검색을 구현하고, 모델별 검색 성능을 비교하는 것을 목표로 합니다.

## 주요 기능

- 현재: CLIP 기반 영어 문장 검색, 유사한 이미지 Top-5 경로와 점수 출력
- 예정: 이미지 기반 검색, 웹 화면, CLIP · SigLIP2 성능 비교

## 개발 환경

- Ubuntu 24.04
- Python 3.12 이상
- uv, PyTorch, Transformers
- NVIDIA RTX 3070 Ti 8GB

## 실행 방법

Git과 uv가 설치된 Ubuntu 터미널에서 저장소를 복제하고 프로젝트 폴더로 이동합니다.

```bash
git clone https://github.com/alanworks72/visual-semantic-search.git
cd visual-semantic-search
```

의존성을 설치하고 검색할 이미지를 넣을 폴더를 생성합니다.

```bash
uv sync --frozen
mkdir -p data/sample
```

`data/sample/`에 JPG, JPEG 또는 PNG 이미지를 넣은 뒤 프로젝트 루트에서 실행합니다.
이미지 데이터는 저장소에 포함되어 있지 않으므로 직접 준비해야 합니다.
처음에는 20장 정도로 시작합니다.

COCO 이미지를 사용하려면 아래 [데이터셋](#데이터셋) 항목을 참고하세요.

```bash
uv run python scripts/run.py
```

영어 검색어를 입력하면 관련 이미지의 경로와 유사도 점수가 출력됩니다.
예: `a dog running`, `a plate of food`

첫 실행에는 모델 다운로드가 필요합니다. GPU를 사용할 수 있으면 GPU로, 그렇지 않으면 CPU로 실행합니다.

## 데이터셋

본 프로젝트의 개발과 검증(평가)용 데이터셋으로 COCO Captions의 val2017을 사용할 예정입니다.

- [COCO 공식 다운로드 페이지](https://cocodataset.org/#download)
- [2017 Val images — val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (5,000장, 약 1GB)
- [2017 Train/Val annotation — annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (약 241MB)

이미지 압축을 풀어 `data/val2017/`에 두고, 검색에 사용할 이미지 약 20장을 `data/sample/`로 복사합니다.
캡션 기반 평가에 사용할 파일은 어노테이션 압축에 포함된 `annotations/captions_val2017.json`입니다.

데이터 다운로드 및 사용에는 [COCO 공식 이용 조건](https://cocodataset.org/#termsofuse)이 적용됩니다.

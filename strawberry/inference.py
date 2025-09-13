# -*- coding: utf-8 -*-
"""
딸기 생육 단계 분류 - 추론 스크립트
원본 Colab 코드 섹션 8, 9 그대로
"""

import os, glob, re, time, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import timm

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 경로 설정 (VSCode용)
# =========================
PUBLIC_DIR = "/root/sehyun/test/public"
SAMPLE_CSV = "/root/sehyun/sample_submission.csv"
MODEL_PATH = "/root/sehyun/model.pth"
ENSEMBLE_PATH = "/root/sehyun/ensemble_models.pth"
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
DEST_ROOT = f"/root/sehyun/strawberry_cls_run_{RUN_STAMP}"  # 출력 폴더

IMG_SIZE = 384

# =========================
# 모델 로드
# =========================
print("모델 로드 중...")
# 앙상블 모델 로드
checkpoint = torch.load(ENSEMBLE_PATH, map_location=DEVICE)
trained = []
CLASSES = checkpoint['classes']
id2name = checkpoint['id2name']
submit_id_map = checkpoint['submit_id_map']

for model_info in checkpoint['models']:
    model_name = model_info['model_name']
    
    try:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=len(CLASSES),
            img_size=IMG_SIZE,
        ).to(DEVICE)
    except TypeError:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=len(CLASSES),
        ).to(DEVICE)
    
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    trained.append(('', model, model_info['accuracy']))  # name은 필요없어서 빈 문자열
    print(f"✅ 모델 로드: {model_name} (Acc: {model_info['accuracy']:.4f})")

# =========================
# 원본 코드 섹션 8 그대로
# =========================
val_tfms = T.Compose([
    T.Resize(int(IMG_SIZE*1.15), interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

def image_paths_recursive(d):
    exts=(".jpg",".jpeg",".png",".bmp",".webp")
    ps=[]
    for e in exts: ps+=glob.glob(os.path.join(d,"**",f"*{e}"), recursive=True)
    return sorted(ps)

def extract_photo_no(fname):
    nums = re.findall(r"\d+", os.path.basename(fname))
    return int(max(nums, key=len)) if nums else None

if os.path.isdir(PUBLIC_DIR):
    public_imgs = image_paths_recursive(PUBLIC_DIR)
else:
    public_imgs = []

if len(public_imgs)==0:
    print(f"ℹ️ PUBLIC 폴더가 없거나 비어있음: {PUBLIC_DIR}")
else:
    rows=[]; tfm = val_tfms
    with torch.no_grad():
        for pth in tqdm(public_imgs, desc="Public Inference"):
            img = Image.open(pth).convert("RGB")
            x = tfm(img).unsqueeze(0).to(DEVICE)
            probs=[]
            for _, mdl, _ in trained[:2]:
                probs.append(torch.softmax(mdl(x), dim=1).cpu().numpy()[0])
            pavg = np.mean(probs, axis=0)
            pred = int(np.argmax(pavg)); conf=float(pavg[pred])
            rows.append({
                "filename": os.path.basename(pth),
                "label": CLASSES[pred].split('_',1)[1] if '_' in CLASSES[pred] else CLASSES[pred]
            })
    pub_df = pd.DataFrame(rows).sort_values(["A","filename"], na_position="last")
    
    os.makedirs(DEST_ROOT, exist_ok=True)
    pub_csv = f"{DEST_ROOT}/public_predictions_all_ENS_{RUN_STAMP}.csv"
    pub_df.to_csv(pub_csv, index=False, encoding="utf-8-sig")
    print("Saved:", pub_csv)

# =========================
# 원본 코드 섹션 9 그대로
# =========================
# 🔧 여기를 너의 샘플 제출 파일 경로로 변경!
SAMPLE_CSV = "/root/sehyun/sample_submission.csv"

assert len(trained) > 0, "먼저 7)에서 모델을 학습해 trained를 만들어 주세요."
top_models = [mdl for _, mdl, _ in trained[:2]]  # 상위 2개 앙상블 사용

# -- 유틸: 이미지 1장 예측
def predict_one_image(img_path, tfm=val_tfms, models=top_models):
    with torch.no_grad():
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(DEVICE)
        probs = []
        for mdl in models:
            mdl.eval()
            probs.append(torch.softmax(mdl(x), dim=1).cpu().numpy()[0])
        pavg = np.mean(probs, axis=0)
        pred_id = int(np.argmax(pavg))
        pred_label = CLASSES[pred_id].split("_", 1)[1] if "_" in CLASSES[pred_id] else CLASSES[pred_id]
        conf = float(pavg[pred_id])
    return pred_id, pred_label, conf

# -- PUBLIC 폴더에서 파일 매칭 도우미
def find_public_image_by_filename(filename):
    cand = glob.glob(os.path.join(PUBLIC_DIR, "**", filename), recursive=True)
    return cand[0] if len(cand) > 0 else None

def find_public_image_by_A(a_value):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    patterns = [f"*{a_value}*{ext}" for ext in exts]
    cands = []
    for pat in patterns:
        cands += glob.glob(os.path.join(PUBLIC_DIR, "**", pat), recursive=True)
    for p in sorted(cands):
        if extract_photo_no(p) == a_value:
            return p
    return None

# -- sample_submission 읽기
sample = pd.read_csv(SAMPLE_CSV)
print("🔎 sample_submission columns:", list(sample.columns))

# 어떤 컬럼에 예측을 써야 하는지 자동 판단: label/target/pred 순서
submit_col = None
for cand in ["label", "target", "pred", "prediction", "y"]:
    if cand in sample.columns:
        submit_col = cand
        break
if submit_col is None:
    submit_col = "label"
    sample[submit_col] = ""

# 매칭 키 판단: filename 또는 A(숫자)
has_filename = "filename" in sample.columns
has_A = "A" in sample.columns
if not has_filename and not has_A:
    raise ValueError("sample_submission에 'filename' 또는 'A' 컬럼이 필요합니다.")

pred_rows = []
missing = 0

print("🧠 PUBLIC 예측으로 sample 채우는 중...")
for idx, row in tqdm(sample.iterrows(), total=len(sample)):
    img_path = None

    # 1) filename 우선 매칭
    if has_filename and pd.notna(row.get("filename", None)):
        img_path = find_public_image_by_filename(str(row["filename"]).strip())

    # 2) 실패 시 A로 매칭
    if img_path is None and has_A and pd.notna(row.get("A", None)):
        try:
            a_val = int(row["A"])
            img_path = find_public_image_by_A(a_val)
        except:
            img_path = None

    if img_path is None:
        missing += 1
        pred_rows.append((idx, None, None, None))
        continue

    pred_id, pred_label, conf = predict_one_image(img_path, tfm=val_tfms, models=top_models)
    pred_rows.append((idx, pred_id, pred_label, conf))

# 🔢 sample 채우기 — "숫자 라벨(1~5)"로 저장
for idx, pid, plabel, conf in pred_rows:
    if pid is None:
        continue
    submit_id = submit_id_map.get(plabel, None)
    if submit_id is None:
        klabel = str(plabel).split('_')[-1]
        submit_id = submit_id_map.get(klabel, pid+1)  # 최후수단
    sample.at[idx, submit_col] = int(submit_id)

# 저장
SUBMIT_PATH = f"/root/sehyun/submission.csv"  # VSCode 경로로 수정
os.makedirs(os.path.dirname(SUBMIT_PATH), exist_ok=True)
sample.to_csv(SUBMIT_PATH, index=False, encoding="utf-8-sig")

print(f"\n✅ 제출 파일 저장 완료: {SUBMIT_PATH}")
print(f"   - 채우지 못한 행 수(missing): {missing}")
print(f"   - 제출 컬럼명: {submit_col}")
print("   - 제출 값 예시:", sample[submit_col].head().tolist())
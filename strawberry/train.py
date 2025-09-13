# -*- coding: utf-8 -*-
"""
딸기 생육 단계 분류 - 원본 Colab 코드 그대로
경로만 VSCode 환경에 맞게 수정
"""

# 📦 필수 라이브러리 (pip install timm 필요)
import os, glob, shutil, random, time, re, math, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
import timm
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1) 경로/클래스/하이퍼파라미터
# =========================
SRC_DIR   = "/root/sehyun/data"   # ← VSCode 경로로 변경
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
DEST_BASE = "/root/sehyun/strawberry_cls"
DEST_ROOT = f"{DEST_BASE}_run_{RUN_STAMP}"
PUBLIC_DIR = "/root/sehyun/test/public"  # ← VSCode 경로로 변경

ordered_classes = ["정식기","출뢰기","개화기","과실비대기","수확기"]
class_to_id = {name:i for i, name in enumerate(ordered_classes)}

# 제출용 숫자 매핑 (고정)
submit_id_map = {"정식기":1, "출뢰기":2, "개화기":3, "과실비대기":4, "수확기":5}

# 분할 비율
TRAIN_RATIO, VAL_RATIO = 0.7, 0.2  # TEST는 나머지 0.1
# 증강 타깃(오버샘플링 금지, 원본만 증강으로 채움)
TARGET_TRAIN = 900

# 학습 설정
IMG_SIZE = 384
BATCH    = 28
EPOCHS   = 12
PATIENCE = 4
LABEL_SMOOTH = 0.03

# 후보 모델 (빠르고 강한 2개)
CANDIDATES = [
    ("tf_efficientnet_b3",            5e-4, 1e-2, 0.2, 0.05),  # (model, lr, weight_decay, dropout, mixup_alpha)
    ("swin_tiny_patch4_window7_224",  1e-4, 5e-3, 0.1, 0.00),
]

print("SRC_DIR:", SRC_DIR)
print("DEST_ROOT:", DEST_ROOT)
print("DEVICE:", DEVICE)

# =========================
# 2) 유틸/IO
# =========================
def all_imgs_flat(dir_path):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    return [x for x in glob.glob(os.path.join(dir_path, "*")) if x.lower().endswith(exts)]

def all_imgs_recursive(dir_path):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    paths=[]
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, "**", f"*{ext}"), recursive=True))
    return paths

def safe_copy(paths, dst_dir, desc):
    os.makedirs(dst_dir, exist_ok=True)
    ok, skip = 0, 0
    for f in tqdm(paths, desc=desc, leave=False):
        try:
            shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f))); ok+=1
        except Exception:
            skip+=1
    return ok, skip

# =========================
# 3) 폴더 생성 & 무작위 분리
# =========================
splits = ["train","val","test"]
for sp in splits:
    for name in ordered_classes:
        os.makedirs(f"{DEST_ROOT}/{sp}/{class_to_id[name]}_{name}", exist_ok=True)

print("\n[분할 복사 시작]")
total_counts = {}
for name in ordered_classes:
    src_cls_dir = f"{SRC_DIR}/{name}"
    if not os.path.isdir(src_cls_dir):
        print(f"[경고] {src_cls_dir} 없음 → 스킵")
        total_counts[name] = {"train":0,"val":0,"test":0}
        continue
    files = all_imgs_flat(src_cls_dir)
    random.shuffle(files)
    n = len(files)
    n_tr = int(n*TRAIN_RATIO)
    n_val= int(n*VAL_RATIO)
    tr, va, te = files[:n_tr], files[n_tr:n_tr+n_val], files[n_tr+n_val:]

    print(f"[{name}] 총 {n}장 → train {len(tr)}, val {len(va)}, test {len(te)}")
    safe_copy(tr, f"{DEST_ROOT}/train/{class_to_id[name]}_{name}", f"Copy {name}→train")
    safe_copy(va, f"{DEST_ROOT}/val/{class_to_id[name]}_{name}",   f"Copy {name}→val")
    safe_copy(te, f"{DEST_ROOT}/test/{class_to_id[name]}_{name}",  f"Copy {name}→test")
    total_counts[name] = {"train":len(tr),"val":len(va),"test":len(te)}

print("\n✅ 데이터 무작위 분리 완료!")
print(total_counts)

# =========================
# 4) 오버샘플링 금지: 원본만 기준으로 정확히 TARGET_TRAIN 맞추는 증강
# =========================
DUP_TAG = "_dup"
AUG_TAGS = ["_hf","_r90","_r180","_r270","_r90_hf","_r180_hf","_r270_hf"]
AUG_PREFIX = "_aug"

def looks_augmented(fname: str) -> bool:
    low = fname.lower()
    if any(tag in low for tag in AUG_TAGS): return True
    base = os.path.basename(low)
    return (DUP_TAG in base) or (AUG_PREFIX in base)

def purge_train_to_only_originals(train_root):
    rm_cnt=0
    cls_dirs = sorted([d for d in glob.glob(os.path.join(train_root,"*")) if os.path.isdir(d)])
    for cls_dir in cls_dirs:
        for p in all_imgs_flat(cls_dir):
            if looks_augmented(p):
                try: os.remove(p); rm_cnt+=1
                except: pass
    print(f"🧹 정리: 복제/증강 {rm_cnt}개 삭제 (원본만 유지)")

def rand_hflip(img):
    if random.random()<0.5: img = ImageOps.mirror(img)
    return img

def rand_rotate(img):
    angle = random.uniform(-25,25)
    return img.rotate(angle, expand=True)

def rand_crop_resize(img, min_scale=0.85):
    w,h = img.size
    scale = random.uniform(min_scale,1.0)
    nw, nh = int(w*scale), int(h*scale)
    if nw<1 or nh<1: return img
    left = random.randint(0, w-nw) if w>nw else 0
    top  = random.randint(0, h-nh) if h>nh else 0
    img_c = img.crop((left, top, left+nw, top+nh))
    return img_c.resize((w,h), Image.BICUBIC)

def rand_color_jitter(img):
    if random.random()<0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.88,1.12))
    if random.random()<0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.88,1.12))
    if random.random()<0.8:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.88,1.12))
    return img

def rand_blur(img):
    if random.random()<0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2,1.0)))
    return img

def rand_noise(img):
    if random.random()<0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0,6.0, arr.shape)
        arr = np.clip(arr+noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img

def strong_augment(img):
    ops = [rand_hflip, rand_rotate, rand_crop_resize, rand_color_jitter, rand_blur, rand_noise]
    random.shuffle(ops)
    for op in ops: img = op(img)
    return img

def save_jpg(img: Image.Image, out_path: str):
    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".jpg",".jpeg"]:
        img.convert("RGB").save(out_path, quality=95, subsampling=0)
    else:
        img.save(out_path)

def ensure_train_exact_target_by_augment_only(train_root, target=TARGET_TRAIN):
    os.makedirs(f"{DEST_ROOT}/train_overflow", exist_ok=True)
    cls_dirs = sorted([d for d in glob.glob(os.path.join(train_root,"*")) if os.path.isdir(d)])
    total_created,total_moved=0,0

    for cls_dir in cls_dirs:
        cls_name = os.path.basename(cls_dir)
        base_files = [p for p in all_imgs_flat(cls_dir) if not looks_augmented(p)]
        cur = len(base_files)

        # 원본이 너무 많으면 target만 남기고 overflow 이동
        if cur>target:
            keep_idx = np.random.choice(cur, size=target, replace=False)
            keep_set = set([base_files[i] for i in keep_idx])
            ov_dir = os.path.join(DEST_ROOT,"train_overflow",cls_name); os.makedirs(ov_dir, exist_ok=True)
            moved=0
            for p in base_files:
                if p not in keep_set:
                    try: shutil.move(p, os.path.join(ov_dir, os.path.basename(p))); moved+=1
                    except: pass
            total_moved+=moved
            print(f"[{cls_name}] 원본 {cur} → {target}개만 사용(overflow {moved})")
            continue

        need = target - cur
        if need==0:
            print(f"[{cls_name}] 원본 {cur}개 → target 충족(증강 없음)")
            continue

        print(f"[{cls_name}] 원본 {cur}개 → 증강 {need}개 생성하여 {target} 맞춤")
        created=0; base_idx=0; uid=0
        pbar = tqdm(total=need, desc=f"Augment {cls_name}", leave=False)
        while created<need and cur>0:
            src = base_files[base_idx % cur]
            base,ext = os.path.splitext(src)
            out_path = f"{base}_aug{uid:06d}{ext}"; uid+=1
            if os.path.exists(out_path):
                base_idx+=1; continue
            try:
                with Image.open(src).convert("RGB") as img:
                    aug = strong_augment(img); save_jpg(aug, out_path)
                created+=1; total_created+=1; pbar.update(1)
            except: pass
            base_idx+=1
        pbar.close()
        final = len(all_imgs_flat(cls_dir))
        print(f"  → 생성 {created}개, 최종 {final}개 (목표 {target})")
    print(f"\n✅ 증강 완료: 새 증강 {total_created}개, overflow 이동 {total_moved}개")

purge_train_to_only_originals(os.path.join(DEST_ROOT,"train"))
ensure_train_exact_target_by_augment_only(os.path.join(DEST_ROOT,"train"), TARGET_TRAIN)

# =========================
# 5) DataLoader 구성
# =========================
train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7,1.0), interpolation=InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(0.15,0.15,0.15),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
val_tfms = T.Compose([
    T.Resize(int(IMG_SIZE*1.15), interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

train_ds = ImageFolder(f"{DEST_ROOT}/train", transform=train_tfms)
val_ds   = ImageFolder(f"{DEST_ROOT}/val",   transform=val_tfms)
test_ds  = ImageFolder(f"{DEST_ROOT}/test",  transform=val_tfms)
assert train_ds.classes == val_ds.classes == test_ds.classes
CLASSES = train_ds.classes; NUM_CLASSES=len(CLASSES)
id2name = {i: c.split('_',1)[1] if '_' in c else c for i,c in enumerate(CLASSES)}

num_workers = 2  # Colab 안정성
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=True)

# =========================
# 6) EarlyStopping / 학습 루틴
# =========================
class EarlyStopping:
    def __init__(self, patience=PATIENCE, mode="max", min_delta=1e-4):
        self.patience=patience; self.mode=mode; self.min_delta=min_delta
        self.best=None; self.count=0; self.stop=False
    def step(self, metric):
        if self.best is None: self.best=metric; return False
        better = metric>self.best+self.min_delta if self.mode=="max" else metric<self.best-self.min_delta
        if better: self.best=metric; self.count=0
        else:
            self.count+=1
            if self.count>=self.patience: self.stop=True
        return self.stop

def evaluate(model, loader, criterion=None, mixup_active=False):
    model.eval(); n=0; correct=0; loss_sum=0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            if criterion is not None and not mixup_active:
                loss_sum += criterion(logits, y).item()*x.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            n += x.size(0)
    top1 = correct/max(n,1)
    loss = loss_sum/max(n,1) if criterion is not None and not mixup_active else None
    return loss, top1

def train_model(model_name, epochs, lr, weight_decay, dropout, mixup_alpha,
                label_smoothing=LABEL_SMOOTH, patience=PATIENCE):
    # ✅ Swin/VIT 계열은 img_size 필요, EfficientNet은 불필요 → 안전 처리
    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=dropout,
            img_size=IMG_SIZE,   # 먼저 시도
        ).to(DEVICE)
    except TypeError:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=dropout,
        ).to(DEVICE)

    if mixup_alpha>0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=0.0,
                         label_smoothing=label_smoothing, num_classes=NUM_CLASSES)
        mix_act=True
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing); mixup_fn=None; mix_act=False
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    es = EarlyStopping(patience=patience, mode="max")

    best_top1=-1.0; best_state=None
    for ep in range(1, epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            if mixup_fn: x,y = mixup_fn(x,y)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
        scheduler.step()
        _, vtop1 = evaluate(model, val_loader, criterion if not mix_act else None, mix_act)
        if vtop1>best_top1:
            best_top1=vtop1
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
        print(f"[{model_name}] Ep {ep}/{epochs}  val_top1={vtop1:.4f}  best={best_top1:.4f}")
        if es.step(vtop1):
            print(f"[{model_name}] Early stopping at epoch {ep}"); break

    if best_state: model.load_state_dict({k: v.to(DEVICE) for k,v in best_state.items()})
    return model, best_top1

# =========================
# 7) 두 모델 학습 + 앙상블
# =========================
trained=[]
for (mn, lr, wd, dr, mx) in CANDIDATES:
    print(f"\n==> Train {mn}")
    m, v = train_model(mn, epochs=EPOCHS, lr=lr, weight_decay=wd, dropout=dr, mixup_alpha=mx)
    trained.append((mn, m, v))
trained = sorted(trained, key=lambda x: x[2], reverse=True)
print("\n[VAL Top-1]", [(n, round(v,4)) for (n,_,v) in trained])

def infer_probs(model, loader):
    model.eval(); probs_list=[]; labels=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            p = torch.softmax(model(x), dim=1).cpu().numpy()
            probs_list.append(p); labels.extend(y.numpy().tolist())
    return np.vstack(probs_list), np.array(labels)

# 상위 2개 앙상블 (테스트)
probs_all=[]; y_true=None
for name, mdl, _ in trained[:2]:
    p, y = infer_probs(mdl, test_loader)
    probs_all.append(p)
    if y_true is None: y_true=y
p_avg = np.mean(probs_all, axis=0)
y_pred = p_avg.argmax(axis=1)
conf   = p_avg.max(axis=1)

acc = accuracy_score(y_true, y_pred)
print("\n[TEST-ENSEMBLE] Accuracy:", acc)
print(classification_report(
    y_true, y_pred,
    target_names=[f"{i}:{CLASSES[i].split('_',1)[1] if '_' in CLASSES[i] else CLASSES[i]}" for i in range(len(CLASSES))],
    digits=4
))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

test_paths = [s[0] for s in test_ds.samples]
df_test = pd.DataFrame({
    "path": test_paths,
    "y_true": y_true,
    "y_pred": y_pred,
    "conf": conf,
    "correct": (y_true==y_pred).astype(int),
}).sort_values(["correct","conf"], ascending=[False, False])

out_csv = f"{DEST_ROOT}/test_results_leaderboard_ENS_{RUN_STAMP}.csv"
df_test.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("Saved:", out_csv)

# =========================
# 8) 모델 저장 (VSCode용 추가)
# =========================
print("\n모델 저장 중...")
# 최고 성능 모델 저장
best_model = trained[0][1]
torch.save({
    'model_state_dict': best_model.state_dict(),
    'model_name': trained[0][0],
    'classes': CLASSES,
    'id2name': id2name,
    'submit_id_map': submit_id_map,
    'val_accuracy': trained[0][2]
}, '/root/sehyun/model.pth')

# 앙상블용 모든 모델 저장
ensemble_models = []
for name, model, acc in trained[:2]:
    ensemble_models.append({
        'model_name': name,
        'model_state_dict': model.state_dict(),
        'accuracy': acc
    })

torch.save({
    'models': ensemble_models,
    'classes': CLASSES,
    'id2name': id2name,
    'submit_id_map': submit_id_map
}, '/root/sehyun/ensemble_models.pth')

print("✅ 모델 저장 완료!")
print("  - /root/sehyun/model.pth (단일 모델)")
print("  - /root/sehyun/ensemble_models.pth (앙상블 모델)")
import os
import glob
import random
import shutil
import uuid
from pathlib import Path
import numpy as np
import cv2


# Face augmentations
def augment_face(img):
    h, w = img.shape[:2]
    out = img.copy()

    # lille rotation
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(
        out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    # kun horisontal flip (p=0.5)
    if np.random.rand() < 0.5:
        out = cv2.flip(out, 1)

    # let zoom/crop eller pad
    zoom = np.random.uniform(0.95, 1.08)
    nw, nh = int(w * zoom), int(h * zoom)
    out2 = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if zoom >= 1.0:
        x0 = (nw - w) // 2
        y0 = (nh - h) // 2
        out = out2[y0 : y0 + h, x0 : x0 + w]
    else:
        x0 = (w - nw) // 2
        y0 = (h - nh) // 2
        out = cv2.copyMakeBorder(
            out2, y0, h - nh - y0, x0, w - nw - x0, borderType=cv2.BORDER_REFLECT_101
        )

    # svag brightness/contrast jitter
    alpha = np.random.uniform(0.9, 1.1)
    beta = np.random.uniform(-8, 8)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # meget svag gaussian noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 3, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def balance_train_folder(
    src_dir, dst_dir, target=5000, exts=(".png", ".jpg", ".jpeg", ".bmp")
):
    src = Path(src_dir)
    dst = Path(dst_dir)

    if not src.is_dir():
        raise FileNotFoundError(
            f"Kildemappe findes ikke: {src}\nArbejdsmappe: {os.getcwd()}"
        )

    classes = [p.name for p in src.iterdir() if p.is_dir()]
    if not classes:
        raise ValueError(f"Ingen klassemapper i {src} (forventet fx angry/happy/...)")

    # Saml filer
    paths = {}
    for c in classes:
        files = []
        for e in exts:
            files += glob.glob(str(src / c / f"*{e}"))
        paths[c] = sorted(files)

    counts = {c: len(v) for c, v in paths.items()}
    if sum(counts.values()) == 0:
        raise ValueError(f"Ingen billeder fundet i {src}. Tjek filendelser: {exts}")

    print("Start counts:", counts)
    print("Target per class:", target)

    dst.mkdir(parents=True, exist_ok=True)

    for c in classes:
        cls_dst = dst / c
        cls_dst.mkdir(parents=True, exist_ok=True)
        files = paths[c]
        n = len(files)

        if n == 0:
            print(f"Advarsel: ingen billeder i '{c}', springer.")
            continue

        # UNDERSAMPLE hvis > target
        if n > target:
            keep = set(random.sample(files, target))
            for p in keep:
                shutil.copy2(p, cls_dst / Path(p).name)
            print(f"{c}: undersampled {n} → {target}")

        # ellers kopier alle og OVERSAMPLE til target
        else:
            for p in files:
                shutil.copy2(p, cls_dst / Path(p).name)
            added = 0
            while n + added < target:
                src_path = random.choice(files)
                img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    # kan ske hvis sti har specialtegn — spring over
                    continue
                aug = augment_face(img)
                out_name = (
                    cls_dst / f"{Path(src_path).stem}_aug_{uuid.uuid4().hex[:8]}.jpg"
                )
                if cv2.imwrite(str(out_name), aug):
                    added += 1
            print(f"{c}: oversampled {n} → {target} (+{added})")

    print(f"Færdig. Output: {dst}")


if __name__ == "__main__":
    balance_train_folder("FER-2013/train", "FER-2013/train_balanced", target=5000)

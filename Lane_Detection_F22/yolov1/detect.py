from pathlib import Path

import torch
from draw_utils import draw_data_loader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
print(ROOT)
import cv2

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# Directories
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Save results (image with detections)
def detect(val_loader, model_, imgs, labels):
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        preds = model_(imgs)
        ground_truth = draw_data_loader(imgs, labels)  # DONE
        # draw_preds, nms = draw_batch(imgs, preds)   #need change change to draw_nms. compute nms inside 
        # tp, fp, fn, tn = accuracy(labels, nms)  #need change
        # TP += tp
        # FP += fp
        # FN += fn
        # TN += tn
        # rec.append(recall(TP, FN))
        # prec.append(precision(TP, FP))

        cv2.imwrite(save_path, im0)
        if batch_idx == 0:
                break
        
    
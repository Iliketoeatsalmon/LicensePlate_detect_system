# lpr_show_topright_lines.py
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from functions.charecter import get_thai_character, data_province, split_license_plate_and_province

# ================== CONFIG ==================
VEHICLE_W = "model/license_plate.pt"
PLATE_W   = "model/data_plate.pt"

# Image folder
IMAGE_DIR = "CarCrops"

# Thai font path (TTF/OTF/TTC). If None or invalid, it will fallback to system fonts.
FONT_PATH = "model/model_data/TH K2D July8.ttf"

# Confidence thresholds
CONF_VEHICLE = 0.30
CONF_CHAR    = 0.35

# Line grouping tolerance (pixels). Increase if camera perspective is strong.
LINE_Y_TOL = 30
# ============================================


# --------- Font & overlay helpers (PIL if available) ---------
def load_font(font_path: str | None, font_size: int):
    """Try to load a Thai font for PIL drawing."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    candidates = []
    if font_path:
        ext = Path(font_path).suffix.lower()
        if ext in [".ttf", ".otf", ".ttc"]:
            candidates.append(font_path)
        else:
            print(f"[WARN] '{font_path}' is not a TTF/OTF/TTC (found {ext}); falling back")

    # Common Thai fonts on macOS (adjust paths for your OS if needed)
    candidates += [
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "/Library/Fonts/Thonburi.ttc",
        "/Library/Fonts/Th Sarabun New.ttf",
        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
        "/Library/Fonts/NotoSansThai-Regular.ttf",
    ]

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return ImageFont.truetype(p, font_size)
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_text_top_right(image_bgr, lines, font_obj=None, font_size=28, pad=12):
    """
    Draw a translucent rounded box + multi-line text at top-right corner.
    Uses PIL if available for proper Thai rendering; otherwise falls back to cv2.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fallback to cv2 (Thai rendering may be imperfect)
        h, w = image_bgr.shape[:2]
        maxw, heights = 0, []
        for t in lines:
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            maxw = max(maxw, tw)
            heights.append(th)
        box_w = maxw + pad*2
        box_h = sum(heights) + pad*(len(lines)+1)
        x0 = max(0, w - box_w - 20)
        y0 = 20
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h), (0,0,0), -1)
        image_bgr = cv2.addWeighted(overlay, 0.35, image_bgr, 0.65, 0.0)
        y = y0 + pad + heights[0]
        for t in lines:
            x = x0 + pad
            cv2.putText(image_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y += th + pad
        return image_bgr

    from PIL import Image, ImageDraw
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    if font_obj is None:
        font_obj = load_font(FONT_PATH, font_size)

    widths, heights = [], []
    if font_obj:
        for t in lines:
            bbox = draw.textbbox((0, 0), t, font=font_obj)  # (l, t, r, b)
            widths.append(bbox[2]-bbox[0])
            heights.append(bbox[3]-bbox[1])
    else:
        for t in lines:
            widths.append(len(t)*font_size//2)
            heights.append(font_size)

    box_w = (max(widths) if widths else 200) + pad*2
    box_h = (sum(heights) if heights else font_size) + pad*(len(lines)+1)

    W, H = pil_img.size
    x0 = max(0, W - box_w - 20)
    y0 = 20
    x1, y1 = x0 + box_w, y0 + box_h

    overlay = Image.new("RGBA", pil_img.size, (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    try:
        odraw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=(0,0,0,160))
    except Exception:
        odraw.rectangle([x0, y0, x1, y1], fill=(0,0,0,160))
    pil_img = Image.alpha_composite(pil_img, overlay)
    draw = ImageDraw.Draw(pil_img)

    y = y0 + pad
    for i, t in enumerate(lines):
        if font_obj:
            draw.text((x0 + pad, y), t, font=font_obj, fill=(255,255,255,255))
            y += heights[i] + pad
        else:
            draw.text((x0 + pad, y), t, fill=(255,255,255,255))
            y += font_size + pad

    return cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


# --------- Plate line grouping & assembly ---------
def group_by_line(tokens, y_tol=LINE_Y_TOL):
    """
    Group character tokens into lines by Y, then sort each line left->right.
    tokens: list of (x1, clsname, (x1,y1,x2,y2))
    return: list of lines (top->bottom), each line is a list of the same token tuples.
    """
    if not tokens:
        return []

    # add y_center then sort by y
    tmp = []
    for x1, clsname, box in tokens:
        y_center = (box[1] + box[3]) / 2.0
        tmp.append((x1, clsname, box, y_center))
    tmp.sort(key=lambda t: t[3])

    lines = []
    for item in tmp:
        placed = False
        for line in lines:
            # compare with current line mean Y
            mean_y = np.mean([it[3] for it in line])
            if abs(item[3] - mean_y) <= y_tol:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    # sort inside each line by x (left->right)
    for line in lines:
        line.sort(key=lambda t: t[0])

    # drop y_center before returning
    out = []
    for line in lines:
        out.append([(x1, cls, box) for (x1, cls, box, yc) in line])
    return out


def assemble_plate_and_province(lines):
    """
    Build final strings from grouped lines.
    - If only 1 line: treat non-province tokens as plate; province from split helper.
    - If 2+ lines (motorcycle common case): pick the line with MOST non-province tokens as 'plate line',
    and the remaining line that has majority province tokens as 'province line'.
    """
    plate_line_tokens = []
    province_line_tokens = []

    if not lines:
        return "", ""

    if len(lines) == 1:
        # Car-like single row
        flat = lines[0]
        non_prov = [cls for (_, cls, _) in flat if cls not in data_province]
        prov = [cls for (_, cls, _) in flat if cls in data_province]
        plate_text = "".join(get_thai_character(c) for c in non_prov)
        joined = "".join(get_thai_character(c) for c in (non_prov + prov))
        _, province_text = split_license_plate_and_province(joined)
        return plate_text, province_text or ""

    # 2 or more lines: choose best candidates
    # score each line by (#non_province, -#province) to prefer plate content
    scored = []
    for idx, line in enumerate(lines):
        non_p = sum(1 for (_, cls, _) in line if cls not in data_province)
        p = sum(1 for (_, cls, _) in line if cls in data_province)
        scored.append((idx, non_p, -p))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    plate_idx = scored[0][0]

    plate_line = lines[plate_idx]
    other_lines = [lines[i] for i in range(len(lines)) if i != plate_idx]

    # province line = the one with most province tokens
    if other_lines:
        other_scores = [(i, sum(1 for (_, cls, _) in ln if cls in data_province))
                        for i, ln in enumerate(other_lines)]
        other_scores.sort(key=lambda x: x[1], reverse=True)
        province_line = other_lines[other_scores[0][0]]
    else:
        province_line = []

    plate_text = "".join(get_thai_character(cls) for (_, cls, _) in plate_line if cls not in data_province)
    province_text = "".join(get_thai_character(cls) for (_, cls, _) in province_line if cls in data_province)

    # As a fallback, also try helper to split province from combined
    if not province_text:
        combined = "".join(get_thai_character(cls) for (_, cls, _) in plate_line + province_line)
        _, province_text2 = split_license_plate_and_province(combined)
        if province_text2:
            province_text = province_text2

    return plate_text, province_text


# --------- Main processing ---------
def process_image(image_path, vehicle_model, plate_model, show=True):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"!! Can't load picture {image_path}")
        return

    annotated = frame.copy()
    vehicle_results = vehicle_model(frame, conf=CONF_VEHICLE, verbose=False)

    # Only demo the first vehicle; remove breaks to process all vehicles in one image.
    plate_text_final, province_text_final = "", ""
    plate_lr_display = ""  # explicit left->right string for overlay
    letters_only, digits_only = "", ""

    for result in vehicle_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Green vehicle bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

            roi = frame[y1:y2, x1:x2]
            plate_results = plate_model(roi, conf=CONF_CHAR, verbose=False)

            tokens = []
            for det in plate_results:
                for pbox in det.boxes:
                    try:
                        if float(pbox.conf[0]) < CONF_CHAR:
                            continue
                    except Exception:
                        pass
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    ax1, ay1, ax2, ay2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
                    cls_id = int(pbox.cls.item()) if hasattr(pbox.cls, "item") else int(pbox.cls)
                    clsname = plate_model.names[cls_id]
                    tokens.append((ax1, clsname, (ax1, ay1, ax2, ay2)))

            # Draw each token bbox (yellow) and group into lines
            for _, _, (cx1, cy1, cx2, cy2) in tokens:
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255,255,0), 2)

            lines = group_by_line(tokens, y_tol=LINE_Y_TOL)

            # Build explicit L->R plate string (concatenate non-province tokens in each line order)
            lr_tokens = []
            for line in lines:                  # top->bottom
                for _, cls, _ in line:          # left->right within a line
                    if cls not in data_province:
                        lr_tokens.append(cls)
            plate_lr_display = "".join(get_thai_character(c) for c in lr_tokens)

            # Build final plate/province using rule set (handles car/motorcycle)
            plate_text_final, province_text_final = assemble_plate_and_province(lines)

            # Optional splits for overlay lines
            letters_only = "".join(ch for ch in plate_text_final if not ch.isdigit())
            digits_only  = "".join(ch for ch in plate_text_final if ch.isdigit())

            break
        break

    # Print to terminal
    print(f"[{os.path.basename(image_path)}] Plate(L->R): {plate_lr_display or '-'} | "
          f"Plate(final): {plate_text_final or '-'} | Province: {province_text_final or '-'}")
    print(f" - Letters: {letters_only or '-'}")
    print(f" - Digits : {digits_only or '-'}")

    # Top-right overlay
    lines_text = [
        f"Plate : {plate_lr_display or '-'}",
        f"Province : {province_text_final or '-'}",
        f"Letters : {letters_only or '-'}",
        f"Digits  : {digits_only or '-'}",
    ]
    font_obj = load_font(FONT_PATH, 28)
    annotated = draw_text_top_right(annotated, lines_text, font_obj=font_obj, font_size=28, pad=12)

    if show:
        cv2.imshow("License Plate", annotated)
        cv2.waitKey(0)     # wait any key to go to next image
        cv2.destroyAllWindows()


def main():
    # Load models
    if not os.path.exists(VEHICLE_W):
        print(f"[ERROR] Model not found: {VEHICLE_W}")
        return
    if not os.path.exists(PLATE_W):
        print(f"[ERROR] Model not found: {PLATE_W}")
        return

    vehicle_model = YOLO(VEHICLE_W)
    plate_model   = YOLO(PLATE_W)

    # Iterate images and wait for a key after each
    files = sorted([f for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not files:
        print(f"[INFO] No images found in {IMAGE_DIR}")
        return

    for fname in files:
        path = os.path.join(IMAGE_DIR, fname)
        print(f"Processing {path}")
        process_image(path, vehicle_model, plate_model, show=True)


if __name__ == "__main__":
    main()

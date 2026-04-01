from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64, io, os, sys
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
#  VIT BMI MODEL — load once at startup
#  Point WEIGHTS_PATH at your aug_epoch_7.pt file
# ═══════════════════════════════════════════════════════════════════

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../face-to-bmi-vit/weights/aug_epoch_7.pt")
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

# Add the face-to-bmi-vit scripts folder so we can import their model definition
VIT_SCRIPTS  = os.path.join(os.path.dirname(__file__), "../face-to-bmi-vit/scripts")
sys.path.insert(0, VIT_SCRIPTS)

try:
    from models import get_model
    _vit_model = get_model().float().to(DEVICE)
    _vit_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    _vit_model.eval()
    print(f"✅ ViT model loaded on {DEVICE}")
    VIT_AVAILABLE = True
except Exception as e:
    print(f"⚠️  ViT model failed to load: {e}")
    print("    Falling back to geometric model.")
    VIT_AVAILABLE = False

# Image transform — must match what the ViT was trained on
_vit_transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_bmi_vit(image_base64: str) -> float | None:
    """Run the ViT model on a base64-encoded JPEG/PNG. Returns BMI float."""
    try:
        img_bytes = base64.b64decode(image_base64)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x         = _vit_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            bmi = _vit_model(x).item()
        return float(np.clip(bmi, 13.0, 55.0))
    except Exception as e:
        print(f"ViT inference error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  MEDIAPIPE FACEMESH LANDMARK INDEX MAP
# ═══════════════════════════════════════════════════════════════════

LM = {
    "forehead_top":          10,
    "chin_bottom":           152,
    "nose_bridge_top":       6,
    "cheek_left":            234,
    "cheek_right":           454,
    "cheek_inner_left":      116,
    "cheek_inner_right":     345,
    "jaw_left":              172,
    "jaw_right":             397,
    "chin_left":             176,
    "chin_right":            400,
    "chin_center":           18,
    "brow_left_inner":       107,
    "brow_left_outer":       46,
    "brow_right_inner":      336,
    "brow_right_outer":      276,
    "brow_mid":              9,
    "eye_left_outer":        33,
    "eye_left_inner":        133,
    "eye_left_top":          159,
    "eye_left_bottom":       145,
    "eye_right_outer":       263,
    "eye_right_inner":       362,
    "eye_right_top":         386,
    "eye_right_bottom":      374,
    "nose_tip":              4,
    "nose_bottom":           2,
    "nostril_left_base":     240,
    "nostril_right_base":    460,
    "mouth_left":            61,
    "mouth_right":           291,
    "upper_lip_top":         13,
    "lower_lip_bottom":      14,
    "temple_left":           162,
    "temple_right":          389,
}


# ═══════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════

def p(pts, name):
    return pts[LM[name]]

def d2(pts, a, b):
    return float(np.linalg.norm(p(pts, a)[:2] - p(pts, b)[:2]))

def nd(pts, a, b, face_h):
    return d2(pts, a, b) / face_h if face_h > 1e-6 else 0.0


# ═══════════════════════════════════════════════════════════════════
#  PER-REGION MEASUREMENTS (kept for facial observations)
# ═══════════════════════════════════════════════════════════════════

def measure_skull(pts):
    face_h   = d2(pts, "forehead_top", "chin_bottom")
    face_w   = d2(pts, "cheek_left",   "cheek_right")
    temple_w = d2(pts, "temple_left",  "temple_right")
    aspect   = face_w / face_h if face_h > 1e-6 else 0.72

    if   aspect < 0.62: shape, notes = "Oblong",        ["Narrow elongated frame — lean phenotype"]
    elif aspect < 0.72: shape, notes = "Oval",          ["Oval frame — balanced proportions"]
    elif aspect < 0.80: shape, notes = "Round",         ["Round frame — fuller body composition indicator"]
    else:               shape, notes = "Wide / Square", ["Wide frame — strong higher-mass indicator"]

    return {
        "face_height":  round(face_h,   6),
        "face_width":   round(face_w,   6),
        "temple_width": round(temple_w, 6),
        "face_aspect":  round(aspect,   4),
        "face_shape":   shape,
    }, notes


def measure_cheek_jaw(pts, face_h):
    cheek_w  = nd(pts, "cheek_left",       "cheek_right",       face_h)
    jaw_w    = nd(pts, "jaw_left",         "jaw_right",         face_h)
    chin_w   = nd(pts, "chin_left",        "chin_right",        face_h)
    inner_w  = nd(pts, "cheek_inner_left", "cheek_inner_right", face_h)

    cjwr           = cheek_w / jaw_w  if jaw_w  > 1e-6 else 1.15
    jaw_chin_ratio = jaw_w   / chin_w if chin_w > 1e-6 else 1.30
    convexity      = cheek_w - inner_w

    notes = []
    if   cjwr < 1.05: notes.append("Jaw nearly as wide as cheeks — square lower face, strong adiposity marker")
    elif cjwr < 1.18: notes.append("Mild jaw taper — average lower face fullness")
    else:             notes.append("Pronounced jaw taper — defined jaw, lean lower face")

    if   convexity > 0.11: notes.append("High cheek convexity — significant outward bulge, fat indicator")
    elif convexity > 0.06: notes.append("Moderate cheek convexity — some cheek fullness")
    else:                  notes.append("Low cheek convexity — flat cheekbones, lean lateral face")

    return {
        "cheek_width_norm":  round(cheek_w,       4),
        "jaw_width_norm":    round(jaw_w,          4),
        "chin_width_norm":   round(chin_w,         4),
        "jaw_taper_ratio":   round(cjwr,           4),
        "jaw_chin_ratio":    round(jaw_chin_ratio, 4),
        "cheek_convexity":   round(convexity,      4),
    }, notes


def measure_fwhr(pts, face_h):
    width  = d2(pts, "cheek_left", "cheek_right")
    height = d2(pts, "brow_mid",   "upper_lip_top")
    fwhr   = width / height if height > 1e-6 else 1.40

    if   fwhr < 1.25: notes = [f"fWHR {round(fwhr,2)} — narrow upper face, lean indicator"]
    elif fwhr < 1.55: notes = [f"fWHR {round(fwhr,2)} — within average MediaPipe range (~1.40)"]
    else:             notes = [f"fWHR {round(fwhr,2)} — wide upper face, higher BMI indicator"]

    return {"fwhr": round(fwhr, 4)}, notes


def measure_nose(pts, face_h):
    alar_w   = nd(pts, "nostril_left_base", "nostril_right_base", face_h)
    nose_h   = nd(pts, "nose_bridge_top",   "nose_bottom",        face_h)
    iod      = nd(pts, "eye_left_inner",    "eye_right_inner",    face_h)
    nose_iod = alar_w / iod if iod > 1e-6 else 1.00

    if   nose_iod > 1.12: notes = ["Wide alar base relative to eye spacing — adiposity indicator"]
    elif nose_iod < 0.88: notes = ["Narrow nasal base — lean facial phenotype"]
    else:                 notes = ["Nasal width proportionate to inter-ocular distance"]

    return {
        "alar_width_norm":  round(alar_w,   4),
        "nose_height_norm": round(nose_h,   4),
        "nose_iod_ratio":   round(nose_iod, 4),
    }, notes


def measure_mouth(pts, face_h):
    mouth_w  = nd(pts, "mouth_left",    "mouth_right",      face_h)
    lower_fh = nd(pts, "nose_bottom",   "chin_bottom",      face_h)
    lip_h    = nd(pts, "upper_lip_top", "lower_lip_bottom", face_h)

    if   lower_fh > 0.37: notes = ["Deep lower face — enlarged chin/jaw, submental fat indicator"]
    elif lower_fh < 0.28: notes = ["Short lower face — compressed chin"]
    else:                  notes = ["Proportionate lower face height"]

    return {
        "mouth_width_norm":  round(mouth_w,  4),
        "lower_face_h_norm": round(lower_fh, 4),
        "lip_height_norm":   round(lip_h,    4),
    }, notes


def measure_forehead(pts, face_h):
    fore_w = nd(pts, "temple_left",      "temple_right",      face_h)
    fore_h = nd(pts, "forehead_top",     "brow_mid",          face_h)
    brow_l = nd(pts, "brow_left_inner",  "brow_left_outer",   face_h)
    brow_r = nd(pts, "brow_right_inner", "brow_right_outer",  face_h)

    if   fore_h > 0.30: notes = ["High forehead — large upper facial third"]
    elif fore_h < 0.20: notes = ["Low forehead — compressed upper third"]
    else:               notes = ["Proportionate forehead height"]

    return {
        "forehead_width_norm":  round(fore_w,                4),
        "forehead_height_norm": round(fore_h,                4),
        "brow_arc_avg_norm":    round((brow_l + brow_r) / 2, 4),
    }, notes


def measure_thirds(pts, face_h):
    upper  = nd(pts, "forehead_top", "brow_mid",    face_h)
    middle = nd(pts, "brow_mid",     "nose_bottom", face_h)
    lower  = nd(pts, "nose_bottom",  "chin_bottom", face_h)

    notes = []
    if   lower > 0.37: notes.append("Enlarged lower third — fuller chin/jaw, submental fat likely")
    elif lower < 0.28: notes.append("Compressed lower third — short chin")
    else:              notes.append("Lower third within classical proportion range")
    if middle > 0.40:  notes.append("Elongated mid face — long nose bridge")

    return {
        "upper_third":  round(upper,  4),
        "middle_third": round(middle, 4),
        "lower_third":  round(lower,  4),
    }, notes


def measure_pose(pts):
    yaw   = abs(float(p(pts, "eye_left_outer")[2]) - float(p(pts, "eye_right_outer")[2]))
    ez    = (float(p(pts, "eye_left_outer")[2]) + float(p(pts, "eye_right_outer")[2])) / 2
    pitch = abs(float(p(pts, "nose_tip")[2]) - ez)

    if   yaw > 0.20 or pitch > 0.25: quality = "Poor"
    elif yaw > 0.10 or pitch > 0.15: quality = "Fair"
    else:                             quality = "Good"

    return {"yaw_offset": round(yaw,4), "pitch_offset": round(pitch,4), "pose_quality": quality}


# ═══════════════════════════════════════════════════════════════════
#  API SCHEMA
# ═══════════════════════════════════════════════════════════════════

class FaceData(BaseModel):
    landmarks:    list
    height:       float
    age:          int  = 30
    sex:          str  = "unknown"
    image_base64: str  = ""       # base64 JPEG from the capture canvas


# ═══════════════════════════════════════════════════════════════════
#  ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(data: FaceData):
    try:
        if not data.landmarks:
            return {"error": "No landmarks received"}
        if len(data.landmarks) < 468:
            return {"error": f"Need ≥468 landmarks, got {len(data.landmarks)}. "
                             f"Ensure refineLandmarks:true in MediaPipe."}

        if isinstance(data.landmarks[0], dict):
            pts = np.array([[l["x"], l["y"], l["z"]] for l in data.landmarks], dtype=np.float32)
        else:
            pts = np.array(data.landmarks, dtype=np.float32)

        face_h = d2(pts, "forehead_top", "chin_bottom")
        if face_h < 1e-5:
            return {"error": "Face too small in frame — move closer to the camera"}

        pose = measure_pose(pts)
        if pose["pose_quality"] == "Poor":
            return {
                "error":        "Head rotation too extreme — please face the camera directly",
                "yaw_offset":   pose["yaw_offset"],
                "pitch_offset": pose["pitch_offset"],
            }

        # ── Facial observations (landmarks) ───────────────────────
        skull,    skull_notes = measure_skull(pts)
        cj,       cj_notes    = measure_cheek_jaw(pts, face_h)
        fwhr_d,   fwhr_notes  = measure_fwhr(pts, face_h)
        nose,     nose_notes  = measure_nose(pts, face_h)
        mouth,    mouth_notes = measure_mouth(pts, face_h)
        forehead, fore_notes  = measure_forehead(pts, face_h)
        thirds,   third_notes = measure_thirds(pts, face_h)

        # ── BMI — ViT if image provided, else geometric fallback ───
        vit_used = False
        if VIT_AVAILABLE and data.image_base64:
            print(f"Running ViT, image_base64 length: {len(data.image_base64)}")
            predicted_bmi = predict_bmi_vit(data.image_base64)
            print(f"ViT predicted BMI: {predicted_bmi}")
            if predicted_bmi is not None:
                vit_used      = True
                predicted_bmi = round(predicted_bmi, 1)
                confidence    = 85   # ViT is consistently more accurate
            else:
                predicted_bmi = None

        if not vit_used:
            # geometric fallback
            POP = {"face_aspect":0.72,"fwhr":1.40,"cjwr":1.15,"cheek_convexity":0.08,"nose_iod":1.00,"lower_third":0.33}
            W   = {"face_aspect":28.0,"fwhr":10.0,"cjwr":-16.0,"cheek_convexity":30.0,"nose_iod":10.0,"lower_third":22.0}
            flat = {"face_aspect":skull["face_aspect"],"fwhr":fwhr_d["fwhr"],"cjwr":cj["jaw_taper_ratio"],
                    "cheek_convexity":cj["cheek_convexity"],"nose_iod":nose["nose_iod_ratio"],"lower_third":thirds["lower_third"]}
            bmi = 24.5
            for key, weight in W.items():
                bmi += weight * (flat[key] - POP[key])
            predicted_bmi = round(float(np.clip(bmi, 13.0, 55.0)), 1)
            pose_penalty  = int(pose["yaw_offset"] * 200 + pose["pitch_offset"] * 150)
            confidence    = max(45, min(91, 85 - pose_penalty))

        # ── Derived stats ──────────────────────────────────────────
        height_m   = data.height / 100.0
        weight_kg  = predicted_bmi * (height_m ** 2)
        weight_lbs = weight_kg * 2.20462

        if   predicted_bmi < 18.5: bmi_cat = "Underweight"
        elif predicted_bmi < 25.0: bmi_cat = "Normal weight"
        elif predicted_bmi < 30.0: bmi_cat = "Overweight"
        else:                      bmi_cat = "Obese"

        sex_coeff = {"male": 10.8, "female": 0.0}.get(data.sex.lower(), 5.4)
        bf_pct    = round(float(np.clip((1.20 * predicted_bmi) + (0.23 * data.age) - sex_coeff - 5.4, 3.0, 60.0)), 1)

        if   predicted_bmi < 22: build = "Slim"
        elif predicted_bmi < 27: build = "Average"
        elif predicted_bmi < 32: build = "Stocky"
        else:                    build = "Heavy"

        all_obs = skull_notes + cj_notes + fwhr_notes + nose_notes + mouth_notes + fore_notes + third_notes

        return {
            "weight_kg":    round(weight_kg,  1),
            "weight_lbs":   round(weight_lbs, 1),
            "bmi":          predicted_bmi,
            "bmi_category": bmi_cat,
            "body_fat_pct": f"~{bf_pct}%",
            "confidence":   confidence,
            "scan_quality": pose["pose_quality"],
            "build":        build,
            "model_used":   "ViT (neural network)" if vit_used else "Geometric fallback",

            "facial_observations": all_obs,
            "health_insights": [
                "Estimate based on facial analysis — lighting and angle affect accuracy.",
                "For clinical measurements consult a healthcare professional.",
            ],

            "measurements": {
                "skull":     skull,
                "cheek_jaw": cj,
                "fwhr":      fwhr_d,
                "nose":      nose,
                "mouth":     mouth,
                "forehead":  forehead,
                "thirds":    thirds,
                "pose":      pose,
            },
            "methodology": "ViT-H fine-tuned on VisualBMI dataset" if vit_used else "Multi-region facial morphometrics",
            "status": "Success",
        }

    except IndexError as e:
        return {"error": f"Landmark index out of range: {e} — ensure refineLandmarks:true"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Internal error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

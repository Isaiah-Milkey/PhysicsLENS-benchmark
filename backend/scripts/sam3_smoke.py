"""De-risk SAM3: segment+track a concept in one video, report instances/frames."""
import sys, time
from pathlib import Path
import cv2, numpy as np, torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

if len(sys.argv) < 2:
    sys.exit("usage: python sam3_smoke.py <video_path> [concept_prompt]")
VID = sys.argv[1]
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "ball"
MAX_FRAMES, TARGET_H = 40, 480

cap = cv2.VideoCapture(VID); frames = []
while True:
    ok, bgr = cap.read()
    if not ok: break
    h, w = bgr.shape[:2]
    if h > TARGET_H: bgr = cv2.resize(bgr, (int(TARGET_H * w / h), TARGET_H))
    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
cap.release()
if len(frames) > MAX_FRAMES:
    idx = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
    frames = [frames[i] for i in idx]
print(f"{len(frames)} frames, prompt='{PROMPT}'", flush=True)

t0 = time.time()
model = Sam3VideoModel.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda:0").eval()
proc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
session = proc.init_video_session(video=frames, inference_device="cuda:0",
                                  video_storage_device="cpu", dtype=torch.bfloat16)
session = proc.add_text_prompt(inference_session=session, text=PROMPT) or session
per_frame = {}
with torch.inference_mode():
    for out in model.propagate_in_video_iterator(inference_session=session,
                                                 max_frame_num_to_track=len(frames)):
        res = proc.postprocess_outputs(session, out)
        fidx = int(getattr(out, "frame_idx", len(per_frame)))
        oids = res["object_ids"].tolist()
        masks = res["masks"].cpu().numpy().astype(bool)
        scores = res["scores"].float().cpu().numpy()
        per_frame[fidx] = {int(o): (int(masks[i].sum()), float(scores[i])) for i, o in enumerate(oids)}

ids = {}
for d in per_frame.values():
    for oid, (area, sc) in d.items():
        ids.setdefault(oid, []).append((area, sc))
print(f"tracked in {time.time()-t0:.1f}s; {len(per_frame)} frames, {len(ids)} instance(s):", flush=True)
for oid, obs in sorted(ids.items()):
    areas = [a for a, _ in obs]; scs = [s for _, s in obs]
    print(f"  obj#{oid}: present {len(obs)}/{len(frames)}f, "
          f"mean_area={np.mean(areas):.0f}px, mean_score={np.mean(scs):.2f}", flush=True)
print("SAM3 OK", flush=True)

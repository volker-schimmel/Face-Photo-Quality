#!/usr/bin/env python3
"""
Passport Photo Compliance Checker — Stable Results (Tkinter)

Fix:
- Prevent flicker in "Live Compliance Results" by:
  * Creating chips once and updating them in place only when values change
  * Throttling chip updates to at most every 200 ms
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading, time, math

# ========================== Tunable thresholds ==========================
HEAD_SIZE_MIN = 0.70
HEAD_SIZE_MAX = 0.80

EYE_BAND_MIN_FROM_BOTTOM = 0.56
EYE_BAND_MAX_FROM_BOTTOM = 0.69

MIN_INTEREYE_PX = 40
MAX_ROLL_DEG    = 5.0
BG_STD_MAX      = 20.0
BG_EDGE_MAX     = 0.12

# ========================== Detectors ==========================
FACE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# ========================== Helpers ==========================
def shade_band(img_bgr, y1, y2, color=(0, 255, 255), alpha=0.25, border_color=(0, 200, 200), border_thickness=2):
    H, W = img_bgr.shape[:2]
    y1 = max(0, min(H-1, int(y1)))
    y2 = max(0, min(H-1, int(y2)))
    if y2 < y1:
        y1, y2 = y2, y1
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, y1), (W-1, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, dst=img_bgr)
    cv2.line(img_bgr, (0, y1), (W-1, y1), border_color, border_thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (0, y2), (W-1, y2), border_color, border_thickness, cv2.LINE_AA)

def status_colors(ok: bool):
    # Soft backgrounds + readable foregrounds
    if ok:
        return {"bg": "#e7f6e7", "fg": "#155d27", "bd": "#b7e2b7", "text": "✅ Pass"}
    else:
        return {"bg": "#fde9e9", "fg": "#7a1c1c", "bd": "#f7c9c9", "text": "❌ Fail"}

# ========================== App ==========================
class PassportPhotoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Passport Photo Compliance Checker")
        self.root.geometry("900x900")

        # Camera vars
        self.camera = None
        self.camera_running = False
        self.camera_thread = None
        self.current_frame = None
        self.processed_image = None

        # Results state for stable UI
        self.result_rows = {}     # label -> {"row":Frame, "chip":Label, "label_widget":Label}
        self.last_results = {}    # label -> "✅"/"❌"
        self.last_render_ts = 0.0 # throttle timestamp
        self.render_interval = 0.2  # seconds (5 Hz)

        # GUI setup
        self.setup_gui()
        self.initialize_camera()

    # ---------------- GUI ----------------
    def setup_gui(self):
        # No left panel by design
        try:
            style = ttk.Style()
            style.theme_use('clam')
        except Exception:
            pass

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Overlay / Measurements (unchanged features)
        self.overlay_frame = ttk.LabelFrame(main, text="🔍 Live Measurements (Overlay)", padding=10)
        self.overlay_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.processed_label = ttk.Label(self.overlay_frame, text="Waiting for camera…")
        self.processed_label.pack(pady=10)

        # Controls under the overlay panel
        controls = ttk.Frame(main)
        controls.pack(side=tk.TOP, fill=tk.X, pady=(10, 6))

        self.start_btn = ttk.Button(controls, text="🎥 Start", command=self.start_camera)
        self.stop_btn  = ttk.Button(controls, text="⏹ Stop",  command=self.stop_camera)
        self.load_btn  = ttk.Button(controls, text="📁 Load still image", command=self.load_image)
        self.save_btn  = ttk.Button(controls, text="💾 Save overlay",   command=self.save_image)

        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.load_btn.pack(side=tk.LEFT, padx=12)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Compliance Results under the overlay panel — CHIP STYLE (stable)
        results_frame = ttk.LabelFrame(main, text="📊 Live Compliance Results", padding=10)
        results_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(6, 0))

        self.results_container = ttk.Frame(results_frame)
        self.results_container.pack(fill=tk.X)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    # ------------- CHIP rendering helpers (stable; no destroy/rebuild) -------------
    def ensure_result_row(self, label_text: str):
        if label_text in self.result_rows:
            return self.result_rows[label_text]

        row = ttk.Frame(self.results_container)
        row.pack(fill=tk.X, pady=3)

        lbl = ttk.Label(row, text=label_text, anchor="w")
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        chip = tk.Label(
            row,
            text="…",
            padx=10, pady=4,
            bg="#f3f3f3", fg="#333333",
            font=("Segoe UI", 10, "bold"),
            relief=tk.SOLID, bd=1
        )
        chip.pack(side=tk.RIGHT)
        ttk.Label(row, text=" ").pack(side=tk.RIGHT, padx=4)

        self.result_rows[label_text] = {"row": row, "chip": chip, "label_widget": lbl}
        return self.result_rows[label_text]

    def render_results(self, checks):
        """
        checks: list of (label, "✅" or "❌")
        - Throttled to reduce UI churn
        - Updates chip widgets in place only if value changed
        """
        now = time.monotonic()
        if (now - self.last_render_ts) < self.render_interval:
            return  # skip this frame to avoid flicker
        self.last_render_ts = now

        for label, status in checks:
            row = self.ensure_result_row(label)
            last = self.last_results.get(label)
            if last == status:
                continue  # no UI change → no repaint

            ok = (status == "✅")
            c = status_colors(ok)
            row["chip"].config(text=c["text"], bg=c["bg"], fg=c["fg"], bd=1)

            # keep data so we only repaint when needed
            self.last_results[label] = status

    # ---------------- Camera ----------------
    def initialize_camera(self, index: int = 0):
        try:
            self.camera = cv2.VideoCapture(index)
            if self.camera.isOpened():
                self.status_var.set(f"Camera initialized (index {index})")
            else:
                self.status_var.set("❌ No camera found")
        except Exception as e:
            self.status_var.set(f"Camera error: {e}")

    def start_camera(self):
        if self.camera and not self.camera_running:
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            self.status_var.set("Camera started (streaming metrics)")

    def stop_camera(self):
        self.camera_running = False
        self.status_var.set("Camera stopped")

    def camera_loop(self):
        while self.camera_running:
            ret, frame_bgr = self.camera.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb

            overlay_rgb, checks = self.process_frame_for_overlay_and_checks(frame_rgb)
            self.processed_image = Image.fromarray(overlay_rgb)

            mid_disp = cv2.resize(overlay_rgb, (700, 525))
            mid_pil  = Image.fromarray(mid_disp)
            mid_photo = ImageTk.PhotoImage(mid_pil)

            def update_ui():
                self.processed_label.config(image=mid_photo, text="")
                self.processed_label.image = mid_photo
                self.render_results(checks)  # stable, throttled updates

            self.root.after(0, update_ui)
            time.sleep(0.03)  # ~33 FPS

    # ---------------- Processing (overlay unchanged) ----------------
    def process_frame_for_overlay_and_checks(self, frame_rgb: np.ndarray):
        img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        t = max(2, H // 240)
        T = max(2, H // 200)
        f = max(0.6, H / 600.0)
        small_f = max(0.5, H / 900.0)

        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
        overlay = img_bgr.copy()

        y_eye_min = int(H * (1.0 - EYE_BAND_MAX_FROM_BOTTOM))
        y_eye_max = int(H * (1.0 - EYE_BAND_MIN_FROM_BOTTOM))
        shade_band(overlay, y_eye_min, y_eye_max, color=(0, 255, 255), alpha=0.28,
                   border_color=(0, 200, 200), border_thickness=T)
        cv2.putText(overlay, "ICAO Eye Band", (10, max(24, y_eye_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, small_f, (0, 200, 200), t, cv2.LINE_AA)

        head_min_y = int((H - H*HEAD_SIZE_MAX) / 2)
        head_max_y = int((H - H*HEAD_SIZE_MIN) / 2 + H*HEAD_SIZE_MIN)
        shade_band(overlay, head_min_y, head_max_y, color=(0, 180, 0), alpha=0.18,
                   border_color=(0, 180, 0), border_thickness=T)
        cv2.putText(overlay, "Target head height 70–80% of image",
                    (10, max(46, head_min_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, small_f, (0, 180, 0), t, cv2.LINE_AA)

        head_ok = eye_band_ok = eyes_ok = intereye_ok = roll_ok = bg_ok = mouth_closed_ok = "❌"

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (80, 255, 80), T, cv2.LINE_AA)

            head_ratio = h / H
            head_ok = "✅" if HEAD_SIZE_MIN <= head_ratio <= HEAD_SIZE_MAX else "❌"
            cv2.putText(overlay, f"Head: {int(head_ratio*100)}% of H",
                        (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, f, (80, 255, 80), T, cv2.LINE_AA)
            cv2.line(overlay, (x, y), (x + w, y), (80, 255, 80), t, cv2.LINE_AA)
            cv2.line(overlay, (x, y + h), (x + w, y + h), (80, 255, 80), t, cv2.LINE_AA)

            roi_gray = gray[y:y + h, x:x + w]
            eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=6)
            eye_centers = [(x + ex + ew // 2, y + ey + eh // 2) for (ex, ey, ew, eh) in eyes]

            if len(eye_centers) >= 2:
                eyes_ok = "✅"
                if len(eye_centers) > 2:
                    dmax, p1, p2 = -1, None, None
                    for i in range(len(eye_centers)):
                        for j in range(i + 1, len(eye_centers)):
                            d = np.linalg.norm(np.array(eye_centers[i]) - np.array(eye_centers[j]))
                            if d > dmax:
                                dmax, p1, p2 = d, eye_centers[i], eye_centers[j]
                else:
                    p1, p2 = eye_centers[0], eye_centers[1]

                cv2.circle(overlay, p1, max(3, H//200), (255, 80, 80), -1, cv2.LINE_AA)
                cv2.circle(overlay, p2, max(3, H//200), (255, 80, 80), -1, cv2.LINE_AA)
                cv2.line(overlay, p1, p2, (255, 80, 80), T, cv2.LINE_AA)

                dpx = int(np.linalg.norm(np.array(p1) - np.array(p2)))
                intereye_ok = "✅" if dpx >= MIN_INTEREYE_PX else "❌"
                cv2.putText(overlay, f"Inter-eye: {dpx}px", (x, y + h + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, f, (255, 80, 80), T, cv2.LINE_AA)

                eyeline_y = int((p1[1] + p2[1]) / 2)
                cv2.line(overlay, (0, eyeline_y), (W-1, eyeline_y), (0, 0, 255), T, cv2.LINE_AA)
                eye_band_ok = "✅" if y_eye_min <= eyeline_y <= y_eye_max else "❌"

                angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                roll_deg = abs(angle_rad * 180.0 / math.pi)
                roll_ok = "✅" if roll_deg <= MAX_ROLL_DEG else "❌"
                cv2.putText(overlay, f"Roll: {roll_deg:.1f} deg", (x, y + h + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, f, (0, 0, 255), T, cv2.LINE_AA)
            else:
                cv2.putText(overlay, "Eyes not reliably detected",
                            (x, y + h + 24), cv2.FONT_HERSHEY_SIMPLEX, f, (0, 0, 255), T, cv2.LINE_AA)

            mouth_roi_gray = roi_gray[h//2:, :]
            smiles = []
            if not SMILE_CASCADE.empty():
                smiles = SMILE_CASCADE.detectMultiScale(mouth_roi_gray, scaleFactor=1.4, minNeighbors=22)
            mouth_closed_ok = "✅" if len(smiles) == 0 else "❌"
            if len(smiles) > 0:
                sx, sy, sw, sh = smiles[0]
                cv2.rectangle(overlay, (x + sx, y + h//2 + sy), (x + sx + sw, y + h//2 + sy + sh),
                              (0, 0, 255), t, cv2.LINE_AA)
                cv2.putText(overlay, "Smile/Mouth open", (x, max(24, y - 28)),
                            cv2.FONT_HERSHEY_SIMPLEX, f, (0, 0, 255), T, cv2.LINE_AA)

            bg_mask = np.ones((H, W), dtype=np.uint8) * 255
            bg_mask[y:y + h, x:x + w] = 0
            bg_pixels = gray[bg_mask == 255]
            bg_std = float(bg_pixels.std()) if bg_pixels.size > 0 else 255.0
            edges = cv2.Canny(gray, 60, 120)
            edge_density = edges[bg_mask == 255].mean() / 255.0 if bg_pixels.size > 0 else 1.0
            bg_ok = "✅" if (bg_std <= BG_STD_MAX and edge_density <= BG_EDGE_MAX) else "❌"

            cv2.putText(overlay, f"BG std: {bg_std:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, small_f, (220, 220, 220), T-1, cv2.LINE_AA)
            cv2.putText(overlay, f"BG edges: {edge_density:.2f}", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, small_f, (220, 220, 220), T-1, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        checks = [
            (f"Head size {int(HEAD_SIZE_MIN*100)}–{int(HEAD_SIZE_MAX*100)}% (of image height)",
             "✅" if (len(faces)>0 and HEAD_SIZE_MIN <= faces[0][3]/H <= HEAD_SIZE_MAX) else "❌"),
            (f"Eye line within band ({int(EYE_BAND_MIN_FROM_BOTTOM*100)}–{int(EYE_BAND_MAX_FROM_BOTTOM*100)}% from bottom)", eye_band_ok),
            ("Eyes detected", eyes_ok),
            (f"Inter-eye ≥ {MIN_INTEREYE_PX}px", intereye_ok),
            (f"Roll ≤ {int(MAX_ROLL_DEG)}°", roll_ok),
            ("Background uniform (low texture/edges)", bg_ok),
            ("Mouth closed / neutral (no smile)", mouth_closed_ok),
        ]

        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), checks

    # ---------------- File I/O ----------------
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg")])
        if not file_path:
            return
        try:
            bgr = cv2.imread(file_path)
            if bgr is None:
                messagebox.showwarning("Load failed", "Unsupported or unreadable image.")
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb, checks = self.process_frame_for_overlay_and_checks(rgb)
            self.processed_image = Image.fromarray(overlay_rgb)

            mid_disp = cv2.resize(overlay_rgb, (700, 525))
            mid_pil  = Image.fromarray(mid_disp)
            mid_photo = ImageTk.PhotoImage(mid_pil)
            self.processed_label.config(image=mid_photo, text="")
            self.processed_label.image = mid_photo

            self.render_results(checks)  # stable, throttled
            self.status_var.set(f"Loaded still image: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def save_image(self):
        if not self.processed_image:
            messagebox.showinfo("Nothing to save", "No processed frame available yet.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                            filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if file_path:
            self.processed_image.save(file_path)
            self.status_var.set(f"Saved overlay: {file_path}")

    # ---------------- Shutdown ----------------
    def on_closing(self):
        self.stop_camera()
        if self.camera:
            self.camera.release()
        self.root.destroy()

# ========================== Run ==========================
if __name__ == "__main__":
    root = tk.Tk()
    app = PassportPhotoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

#!/usr/bin/env python3
"""
Garbage Detection Dashboard v2 – YOLOv8n Integration
Dual cameras | Manual robot control | Servo | Line following | Settings
+ Real-time garbage detection using YOLOv8n (COCO garbage classes only)
"""

import cv2
import time
import threading
import json
import os
import subprocess
import numpy as np
from datetime import datetime
from collections import deque
from flask import Flask, Response, render_template_string, jsonify, request

# ── Hardware imports (graceful fallback on dev/Windows machines) ───────────────
try:
    from picamera2 import Picamera2
    import libcamera
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

try:
    import YB_Pcb_Car as _YB
    HAS_CAR = True
except ImportError:
    HAS_CAR = False

# ── GPIO pin map for line tracking sensors ────────────────────────────────────
PIN_L1, PIN_L2 = 13, 15   # Left  outer / inner
PIN_R1, PIN_R2 = 11,  7   # Right inner / outer
PIN_TRIG, PIN_ECHO = 16, 18  # Ultrasonic sensor pins

# ── Detection class names ─────────────────────────────────────────────────────
ALL_CLASSES = ["battery","can","cardboard","drink carton","glass bottle",
               "paper","plastic bag","plastic bottle","plastic bottle cap","pop tab"]

# ── Default settings ──────────────────────────────────────────────────────────
DEFAULTS = {
    "robot_speed":        70,
    "track_forward":      70,
    "track_right":        60,
    "track_left":         60,
    "servo1_angle":       90,
    "servo2_angle":       90,
    "cam_quality":        80,
    "detect_confidence":  70,
    "detect_cam0":         0,
    "detect_cam1":         0,
    "stop_on_detect":      0,   # Stop robot 5s on detection
    "obstacle_avoid":      0,   # Ultrasonic obstacle avoidance
}
# Per-class enable and per-class confidence
for _cls in ALL_CLASSES:
    _key = _cls.replace(' ','_')
    DEFAULTS[f"cls_en_{_key}"]   = 1    # enabled by default
    DEFAULTS[f"cls_conf_{_key}"] = 70   # per-class confidence
settings = dict(DEFAULTS)


# =============================================================================
# Shared Logger
# =============================================================================
class Logger:
    def __init__(self, maxlen=300):
        self._q = deque(maxlen=maxlen)

    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._q.append({"time": ts, "level": level, "msg": str(msg)})
        print(f"[{ts}] [{level}] {msg}")

    def clear(self):
        self._q.clear()
        self.log("Logs cleared")

    @property
    def entries(self):
        return list(self._q)


log = Logger()


# =============================================================================
# Garbage Detector  (YOLOv8n – COCO garbage classes only)
# =============================================================================
class GarbageDetector:
    GARBAGE_CLASSES = {i: name for i, name in enumerate(ALL_CLASSES)}

    def __init__(self):
        self._model   = None
        self._lock    = threading.Lock()
        self._queue   = {}
        self._results = {}
        self._recent  = deque(maxlen=30)
        self._total   = 0
        self._running = False
        # Records: auto-snapshot per detected object
        self._records      = []   # [{"id","label","conf","time","file"}]
        self._record_id    = 0
        self._seen_objects  = {}   # track unique objects by position hash
        self._stop_until   = 0    # timestamp until robot should stay stopped
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")
            if not os.path.exists(model_path):
                model_path = "yolov8n.pt"
                log.log(f"Custom model not found, falling back to COCO", "WARN")
            self._model   = YOLO(model_path)
            self._running = True
            threading.Thread(target=self._infer_loop, daemon=True).start()
            log.log(f"Stage 2 YOLOv8n detector ready ({model_path})")
        except ImportError:
            log.log("ultralytics not installed – detection disabled", "WARN")
        except Exception as e:
            log.log(f"YOLO load failed: {e}", "ERROR")

    @property
    def available(self): return self._model is not None

    @property
    def should_stop(self): return time.time() < self._stop_until

    def submit(self, cam_id, frame):
        with self._lock:
            self._queue[cam_id] = frame.copy()

    def _is_class_enabled(self, label):
        key = label.replace(' ','_')
        return bool(settings.get(f"cls_en_{key}", 1))

    def _class_conf(self, label):
        key = label.replace(' ','_')
        return settings.get(f"cls_conf_{key}", 70) / 100.0

    def _obj_hash(self, cam_id, label, x1, y1, x2, y2):
        # Quantize position to grid cells to avoid re-snapping same object
        cx, cy = (x1+x2)//2 // 80, (y1+y2)//2 // 80
        return f"{cam_id}_{label}_{cx}_{cy}"

    def _auto_record(self, cam_id, frame, label, conf, x1, y1, x2, y2):
        ohash = self._obj_hash(cam_id, label, x1, y1, x2, y2)
        now = time.time()
        # Skip if we already recorded this object recently (within 30s)
        if ohash in self._seen_objects and (now - self._seen_objects[ohash]) < 30:
            return
        self._seen_objects[ohash] = now
        # Save cropped snapshot
        rec_dir = os.path.join(os.path.dirname(__file__), "records")
        os.makedirs(rec_dir, exist_ok=True)
        self._record_id += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rec_{self._record_id}_{label.replace(' ','_')}_{ts}.jpg"
        fpath = os.path.join(rec_dir, fname)
        try:
            h, w = frame.shape[:2]
            pad = 20
            crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
            cv2.imwrite(fpath, crop)
        except Exception:
            cv2.imwrite(fpath, frame)
        rec = {"id": self._record_id, "label": label, "conf": round(conf*100),
               "time": datetime.now().strftime("%H:%M:%S"), "file": fname}
        self._records.append(rec)
        log.log(f"📸 Record #{self._record_id}: {label} ({conf:.0%}) saved")
        # Stop-on-detect
        if settings.get("stop_on_detect", 0):
            self._stop_until = time.time() + 5
            log.log("🛑 Robot paused 5s (object detected)")

    def _infer_loop(self):
        while self._running:
            with self._lock:
                items = list(self._queue.items())
                self._queue.clear()
            if not items:
                time.sleep(0.05)
                continue
            global_conf = settings.get("detect_confidence", 70) / 100.0
            for cam_id, frame in items:
                if not settings.get(f"detect_cam{cam_id}", 0):
                    with self._lock: self._results[cam_id] = []
                    continue
                try:
                    results = self._model(frame, verbose=False, conf=global_conf)[0]
                    dets = []
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in self.GARBAGE_CLASSES: continue
                        label = self.GARBAGE_CLASSES[cls_id]
                        if not self._is_class_enabled(label): continue
                        c = float(box.conf[0])
                        if c < self._class_conf(label): continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        dets.append((label, c, x1, y1, x2, y2))
                        self._recent.append(label)
                        self._total += 1
                        self._auto_record(cam_id, frame, label, c, x1, y1, x2, y2)
                    with self._lock: self._results[cam_id] = dets
                    if dets:
                        log.log(f"Cam{cam_id}: {', '.join(d[0] for d in dets)}")
                except Exception as e:
                    log.log(f"YOLO cam{cam_id} error: {e}", "ERROR")

    def get_results(self, cam_id):
        with self._lock: return list(self._results.get(cam_id, []))

    def draw(self, frame, cam_id):
        if not settings.get(f"detect_cam{cam_id}", 0): return frame
        dets = self.get_results(cam_id)
        if not dets: return frame
        out = frame.copy()
        for label, conf, x1, y1, x2, y2 in dets:
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,80), 2)
            text = f"{label} {conf:.0%}"
            ty = max(y1-6, 14)
            (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1,ty-th-4), (x1+tw+4,ty+2), (0,0,0), cv2.FILLED)
            cv2.putText(out, text, (x1+2,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,80), 1, cv2.LINE_AA)
        return out

    @property
    def status(self):
        with self._lock:
            return {
                "available": self.available,
                "total": self._total,
                "recent": list(self._recent)[-10:],
                "cam0_dets": [{"label":d[0],"conf":round(d[1]*100)} for d in self._results.get(0,[])],
                "cam1_dets": [{"label":d[0],"conf":round(d[1]*100)} for d in self._results.get(1,[])],
                "records": self._records[-50:],
                "stop_active": self.should_stop,
            }


# =============================================================================
# Ultrasonic Obstacle Avoidance
# =============================================================================
class ObstacleAvoidance:
    def __init__(self):
        self._running = False
        self._alarm   = False
        self._alarm_until = 0
        self._distance = -1
        self._setup_done = False

    def start(self):
        if self._running: return
        if not HAS_GPIO:
            log.log("GPIO unavailable – obstacle avoidance disabled", "WARN")
            return
        if not self._setup_done:
            try:
                GPIO.setup(PIN_TRIG, GPIO.OUT)
                GPIO.setup(PIN_ECHO, GPIO.IN)
                self._setup_done = True
            except Exception as e:
                log.log(f"Ultrasonic GPIO setup failed: {e}", "ERROR")
                return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        log.log("🔊 Ultrasonic obstacle avoidance STARTED")

    def stop(self):
        self._running = False
        self._alarm = False
        log.log("🔇 Ultrasonic obstacle avoidance STOPPED")

    def _measure(self):
        try:
            GPIO.output(PIN_TRIG, True)
            time.sleep(0.00001)
            GPIO.output(PIN_TRIG, False)
            t0 = time.time()
            timeout = t0 + 0.04
            while GPIO.input(PIN_ECHO) == 0 and time.time() < timeout: pass
            t1 = time.time()
            while GPIO.input(PIN_ECHO) == 1 and time.time() < timeout: pass
            t2 = time.time()
            return (t2 - t1) * 17150  # cm
        except Exception:
            return -1

    def _loop(self):
        while self._running:
            if not settings.get("obstacle_avoid", 0):
                time.sleep(0.5)
                continue
            dist = self._measure()
            self._distance = round(dist, 1) if dist > 0 else -1
            if 0 < dist < 20:  # obstacle within 20cm
                if not self._alarm:
                    self._alarm = True
                    self._alarm_until = time.time() + 5
                    log.log(f"⚠️ OBSTACLE at {dist:.0f}cm! Alarm 5s", "WARN")
            if self._alarm and time.time() > self._alarm_until:
                self._alarm = False
            time.sleep(0.15)

    @property
    def info(self):
        return {"running": self._running, "alarm": self._alarm,
                "distance": self._distance}


# =============================================================================
# Camera Handler  (one instance per physical camera)
# =============================================================================
_offline_jpeg = None


def offline_frame():
    """Return a static JPEG placeholder shown when a camera is closed."""
    global _offline_jpeg
    if _offline_jpeg is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (20, 28, 44)
        cv2.putText(img, "CAMERA OFFLINE", (140, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (70, 70, 70), 2)
        cv2.putText(img, "Press  [Open]  to activate",
                    (148, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (55, 55, 55), 1)
        _, jpeg = cv2.imencode('.jpg', img)
        _offline_jpeg = jpeg.tobytes()
    return _offline_jpeg


class CameraHandler:
    """Supports both CSI cameras (Picamera2) and USB cameras (OpenCV)."""

    def __init__(self, cam_id, cam_type='csi', usb_index=0):
        self.id       = cam_id
        self.type     = cam_type   # 'csi' or 'usb'
        self._usb_idx = usb_index
        self._cam     = None       # Picamera2 instance (CSI)
        self._cap     = None       # cv2.VideoCapture instance (USB)
        self._lock    = threading.Lock()
        self.frame    = None
        self.running  = False
        self.fps      = 0
        self.frames   = 0
        self._fc      = 0
        self._ft      = time.time()
        self._det_fc  = 0          # frame counter for detection throttle

    # ── open ──────────────────────────────────────────────────────────────────
    def open(self):
        if self.running:
            log.log(f"Cam{self.id}: already open", "WARN")
            return True
        if self.type == 'usb':
            return self._open_usb()
        return self._open_csi()

    def _open_csi(self):
        if not HAS_PICAMERA:
            log.log(f"Cam{self.id}: picamera2 not available", "ERROR")
            return False
        try:
            self._cam = Picamera2(camera_num=self.id)
            cfg = self._cam.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)})
            cfg["transform"] = libcamera.Transform(hflip=1, vflip=1)
            self._cam.configure(cfg)
            self._cam.start()
            self.running = True
            threading.Thread(target=self._loop_csi, daemon=True).start()
            log.log(f"Cam{self.id}: opened (CSI)")
            return True
        except Exception as e:
            log.log(f"Cam{self.id}: CSI open failed – {e}", "ERROR")
            self._cam = None
            return False

    def _open_usb(self):
        try:
            cap = cv2.VideoCapture(self._usb_idx)
            if not cap.isOpened():
                log.log(f"Cam{self.id}: USB /dev/video{self._usb_idx} not found", "ERROR")
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap    = cap
            self.running = True
            threading.Thread(target=self._loop_usb, daemon=True).start()
            log.log(f"Cam{self.id}: opened (USB /dev/video{self._usb_idx})")
            return True
        except Exception as e:
            log.log(f"Cam{self.id}: USB open failed – {e}", "ERROR")
            return False

    # ── close ─────────────────────────────────────────────────────────────────
    def close(self):
        if not self.running:
            log.log(f"Cam{self.id}: already closed", "WARN")
            return
        self.running = False
        time.sleep(0.15)
        if self.type == 'usb':
            if self._cap:
                self._cap.release()
                self._cap = None
        else:
            with self._lock:
                try:
                    if self._cam:
                        self._cam.stop()
                        self._cam.close()
                except Exception as e:
                    log.log(f"Cam{self.id}: close error – {e}", "ERROR")
                self._cam = None
        self.frame = None
        self.fps   = 0
        log.log(f"Cam{self.id}: closed")

    # ── capture loops ─────────────────────────────────────────────────────────
    def _loop_csi(self):
        while self.running:
            try:
                quality = settings.get("cam_quality", 80)
                with self._lock:
                    if not self._cam:
                        break
                    raw = self._cam.capture_array()
                # Picamera2 RGB888 is BGR in memory — no conversion needed
                # Submit to detector every 6 frames (non-blocking)
                self._det_fc += 1
                if (self._det_fc % 6 == 0
                        and detector is not None
                        and detector.available
                        and settings.get(f"detect_cam{self.id}", 0)):
                    detector.submit(self.id, raw)
                # Draw detection overlay (no-op if detection disabled / no results)
                display = (detector.draw(raw, self.id)
                           if detector is not None and detector.available
                           else raw)
                _, jpeg = cv2.imencode('.jpg', display,
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                self._store(jpeg)
            except Exception as e:
                log.log(f"Cam{self.id}: CSI error – {e}", "ERROR")
                time.sleep(0.5)

    def _loop_usb(self):
        while self.running:
            try:
                if not self._cap:
                    break
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                quality = settings.get("cam_quality", 80)
                # OpenCV gives BGR frames — encode directly
                self._det_fc += 1
                if (self._det_fc % 6 == 0
                        and detector is not None
                        and detector.available
                        and settings.get(f"detect_cam{self.id}", 0)):
                    detector.submit(self.id, frame)
                display = (detector.draw(frame, self.id)
                           if detector is not None and detector.available
                           else frame)
                _, jpeg = cv2.imencode('.jpg', display,
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                self._store(jpeg)
            except Exception as e:
                log.log(f"Cam{self.id}: USB error – {e}", "ERROR")
                time.sleep(0.5)

    def _store(self, jpeg):
        self.frame   = jpeg.tobytes()
        self.frames += 1
        self._fc    += 1
        now = time.time()
        if now - self._ft >= 1.0:
            self.fps = self._fc
            self._fc = 0
            self._ft = now

    # ── MJPEG generator ───────────────────────────────────────────────────────
    def stream(self):
        while True:
            data = self.frame if (self.running and self.frame) else offline_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
            time.sleep(0.033 if self.running else 0.5)

    @property
    def info(self):
        return {"open": self.running, "fps": self.fps, "frames": self.frames,
                "type": self.type}


# =============================================================================
# Robot Controller
# =============================================================================
class Robot:
    def __init__(self):
        self._car     = None
        self._trk     = False
        self._trk_thr = None
        self._gpio_ok = False
        self._boot()

    def _boot(self):
        if HAS_CAR:
            try:
                self._car = _YB.YB_Pcb_Car()
                log.log("Robot hardware ready")
            except Exception as e:
                log.log(f"Robot init failed: {e}", "ERROR")
        else:
            log.log("YB_Pcb_Car not installed – robot disabled", "WARN")

        if HAS_GPIO:
            self._release_gpio()
            try:
                GPIO.setmode(GPIO.BOARD)
                GPIO.setwarnings(False)
                for p in (PIN_L1, PIN_L2, PIN_R1, PIN_R2):
                    GPIO.setup(p, GPIO.IN)
                self._gpio_ok = True
                log.log("GPIO ready for line tracking")
            except Exception as e:
                log.log(f"GPIO init failed: {e}", "ERROR")
        else:
            log.log("RPi.GPIO not installed – tracking disabled", "WARN")

    def _release_gpio(self):
        """Kill any other process holding /dev/gpiomem, then run GPIO.cleanup()."""
        candidates = ['/dev/gpiomem'] + [f'/dev/gpiochip{i}' for i in range(5)]
        devices    = [d for d in candidates if os.path.exists(d)]
        our_pid    = str(os.getpid())
        killed     = False
        for dev in devices:
            try:
                out = subprocess.check_output(
                    ['sudo', 'lsof', '-t', dev],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                for pid in out.split():
                    if pid != our_pid:
                        subprocess.run(['sudo', 'kill', '-9', pid],
                                       stderr=subprocess.DEVNULL)
                        log.log(f"Killed GPIO holder PID {pid} ({dev})", "WARN")
                        killed = True
            except Exception:
                pass
        if killed:
            time.sleep(0.4)
        try:
            GPIO.cleanup()
        except Exception:
            pass

    # ── movement ──────────────────────────────────────────────────────────────
    def move(self, direction, speed=None):
        c = self._car
        if not c:
            log.log("Move ignored – no hardware", "WARN")
            return
        spd = int(speed) if speed is not None else int(settings["robot_speed"])
        actions = {
            "forward":    lambda: c.Car_Run(spd, spd),
            "backward":   lambda: c.Car_Back(spd, spd),
            "left":       lambda: c.Car_Left(spd, spd),
            "right":      lambda: c.Car_Right(spd, spd),
            "spin_left":  lambda: c.Car_Spin_Left(spd, spd),
            "spin_right": lambda: c.Car_Spin_Right(spd, spd),
        }
        fn = actions.get(direction)
        if fn:
            fn()
            log.log(f"Move → {direction} @ {spd}")
        else:
            log.log(f"Unknown direction: {direction}", "WARN")

    def stop(self):
        if self._car:
            self._car.Car_Stop()
        log.log("Stop")

    def servo(self, sid, angle):
        angle = max(0, min(180, int(angle)))
        if self._car:
            self._car.Ctrl_Servo(int(sid), angle)
        log.log(f"Servo {sid} → {angle}°")

    # ── line following ────────────────────────────────────────────────────────
    def start_tracking(self):
        if self._trk:
            log.log("Tracking already running", "WARN")
            return
        if not (self._gpio_ok and self._car):
            log.log("Tracking unavailable – GPIO or car missing", "ERROR")
            return
        self._trk = True
        self._trk_thr = threading.Thread(target=self._track_loop, daemon=True)
        self._trk_thr.start()
        log.log("Line following STARTED")

    def stop_tracking(self):
        if not self._trk:
            return
        self._trk = False
        self.stop()
        log.log("Line following STOPPED")

    def _track_loop(self):
        c = self._car
        while self._trk:
            try:
                fwd = int(settings["track_forward"])
                rsp = int(settings["track_right"])
                lsp = int(settings["track_left"])
                L1  = GPIO.input(PIN_L1)
                L2  = GPIO.input(PIN_L2)
                R1  = GPIO.input(PIN_R1)
                R2  = GPIO.input(PIN_R2)
                # LOW (False) = black line detected
                if   (not L1 or not L2) and not R2:
                    c.Car_Spin_Right(rsp, rsp); time.sleep(0.2)
                elif not L1 and (not R1 or not R2):
                    c.Car_Spin_Left(lsp, lsp);  time.sleep(0.2)
                elif not L1:
                    c.Car_Spin_Left(lsp, lsp);  time.sleep(0.05)
                elif not R2:
                    c.Car_Spin_Right(rsp, rsp); time.sleep(0.05)
                elif not L2 and R1:
                    c.Car_Spin_Left(lsp, lsp);  time.sleep(0.02)
                elif L2 and not R1:
                    c.Car_Spin_Right(rsp, rsp); time.sleep(0.02)
                elif not L2 and not R1:
                    c.Car_Run(fwd, fwd)
                # Check stop-on-detect
                if detector.should_stop:
                    c.Car_Stop()
                    log.log("Paused for detected object...")
                    while detector.should_stop and self._trk:
                        time.sleep(0.2)
                    if self._trk:
                        log.log("Resuming line following")
                # Check obstacle avoidance
                if obstacle._alarm:
                    c.Car_Stop()
                    log.log("Obstacle alarm! Waiting...")
                    while obstacle._alarm and self._trk:
                        time.sleep(0.2)
                    if self._trk:
                        log.log("Obstacle cleared, resuming")
            except Exception as e:
                log.log(f"Track error: {e}", "ERROR")
                time.sleep(0.1)

    @property
    def tracking(self):
        return self._trk

    @property
    def sensor_states(self):
        if not self._gpio_ok:
            return None
        try:
            return {
                "L1": bool(GPIO.input(PIN_L1)),
                "L2": bool(GPIO.input(PIN_L2)),
                "R1": bool(GPIO.input(PIN_R1)),
                "R2": bool(GPIO.input(PIN_R2)),
            }
        except Exception:
            return None

    def cleanup(self):
        self.stop_tracking()
        self.stop()
        if HAS_GPIO:
            try:
                GPIO.cleanup()
            except Exception:
                pass


# =============================================================================
# Flask App  –  global instances
# =============================================================================
# Camera 0 = CSI (Picamera2)
# Camera 1 = USB (OpenCV) — /dev/video8 found by test_usb_camera.py
cameras  = [CameraHandler(0, cam_type='csi'),
            CameraHandler(1, cam_type='usb', usb_index=8)]
robot    = Robot()
detector = GarbageDetector()
obstacle = ObstacleAvoidance()
app      = Flask(__name__)


# =============================================================================
# HTML Template
# =============================================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Garbage Detection Dashboard v2</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:      #0a0e17;
  --bg2:     #111827;
  --card:    #1a2234;
  --raised:  #243049;
  --accent:  #22d3ee;
  --glow:    rgba(34,211,238,.15);
  --pink:    #f472b6;
  --green:   #34d399;
  --amber:   #fbbf24;
  --red:     #f87171;
  --text:    #e2e8f0;
  --text2:   #94a3b8;
  --muted:   #64748b;
  --border:  #1e293b;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family:'Outfit',sans-serif;
  background:var(--bg);
  color:var(--text);
  height:100vh;
  overflow:hidden;
}
body::before {
  content:''; position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(34,211,238,.025) 1px, transparent 1px),
    linear-gradient(90deg,rgba(34,211,238,.025) 1px, transparent 1px);
  background-size:40px 40px; pointer-events:none; z-index:0;
}
.app { position:relative; z-index:1; display:flex; flex-direction:column; height:100vh; }

/* ── Header ──────────────────────────────────────────────────────────────── */
.header {
  display:flex; align-items:center; justify-content:space-between;
  padding:11px 22px; border-bottom:1px solid var(--border);
  background:rgba(17,24,39,.9); backdrop-filter:blur(12px); flex-shrink:0;
}
.logo { display:flex; align-items:center; gap:10px; }
.logo-icon {
  width:34px; height:34px; border-radius:9px;
  background:linear-gradient(135deg,#22d3ee,#06b6d4);
  display:flex; align-items:center; justify-content:center;
  font-size:18px; box-shadow:0 0 20px var(--glow);
}
.logo h1 { font-size:17px; font-weight:700; letter-spacing:-.3px; }
.logo h1 span { color:var(--accent); }

.nav { display:flex; gap:3px; }
.nav-btn {
  padding:7px 15px; border:1px solid transparent; border-radius:8px;
  font-family:'Outfit',sans-serif; font-size:13px; font-weight:600;
  cursor:pointer; background:transparent; color:var(--text2); transition:all .2s;
}
.nav-btn:hover { background:var(--raised); color:var(--text); }
.nav-btn.active {
  background:var(--glow); color:var(--accent);
  border-color:rgba(34,211,238,.3);
}

.hdr-stats {
  display:flex; gap:14px;
  font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--text2);
}
.sv { color:var(--accent); font-weight:600; }
.sv.off { color:var(--muted); }
.sv.tracking { color:var(--green); animation:blink 1.2s ease-in-out infinite; }
.sv.detecting { color:var(--amber); animation:blink 1.2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.5;} }

/* ── Main layout ─────────────────────────────────────────────────────────── */
.main {
  display:grid; grid-template-columns:1fr 340px;
  gap:14px; padding:14px 22px; flex:1; overflow:hidden;
}
.content-area { overflow-y:auto; display:flex; flex-direction:column; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.tab { display:none; flex-direction:column; gap:12px; padding-bottom:8px; }
.tab.active { display:flex; }

/* ── Camera cards ────────────────────────────────────────────────────────── */
.cam-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.cam-card {
  background:var(--card); border-radius:13px;
  border:1px solid var(--border); overflow:hidden;
  display:flex; flex-direction:column;
}
.cam-card-header {
  display:flex; align-items:center; justify-content:space-between;
  padding:9px 13px; background:var(--bg2); border-bottom:1px solid var(--border);
}
.cam-title { display:flex; align-items:center; gap:7px; font-size:13px; font-weight:600; }
.cam-dot { width:8px; height:8px; border-radius:50%; background:var(--muted); }
.cam-dot.on  { background:var(--green); animation:pulse-g 1.5s ease-in-out infinite; }
@keyframes pulse-g {
  0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(52,211,153,.5);}
  50%{opacity:.8;box-shadow:0 0 0 5px rgba(52,211,153,0);}
}
.cam-fps { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text2); }
.btn-cam-toggle {
  padding:4px 11px; border-radius:6px; border:1px solid var(--border);
  font-size:11px; font-weight:700; font-family:'Outfit',sans-serif;
  cursor:pointer; transition:all .2s;
}
.btn-cam-toggle.is-open {
  background:rgba(248,113,113,.1); color:var(--red);
  border-color:rgba(248,113,113,.3);
}
.btn-cam-toggle.is-open:hover { background:rgba(248,113,113,.2); }
.btn-cam-toggle.is-closed {
  background:rgba(52,211,153,.1); color:var(--green);
  border-color:rgba(52,211,153,.3);
}
.btn-cam-toggle.is-closed:hover { background:rgba(52,211,153,.2); }

.cam-view { position:relative; aspect-ratio:4/3; background:#0d1420; }
.cam-view img { width:100%; height:100%; object-fit:contain; display:block; }
.cam-footer {
  padding:6px 13px; font-family:'JetBrains Mono',monospace;
  font-size:10.5px; color:var(--muted); display:flex; gap:14px;
}
/* Detection badge on camera footer */
.det-badge {
  margin-left:auto; padding:1px 8px; border-radius:4px; font-size:10px;
  background:rgba(34,211,238,.12); color:var(--accent); font-weight:700;
  display:none;
}
.det-badge.active { display:inline; }

/* ── Controls tab ────────────────────────────────────────────────────────── */
.ctrl-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.ctrl-card {
  background:var(--card); border-radius:13px;
  border:1px solid var(--border); padding:15px;
}
.ctrl-card h3 {
  font-size:12px; font-weight:700; color:var(--text2); text-transform:uppercase;
  letter-spacing:.5px; margin-bottom:13px;
  display:flex; align-items:center; gap:7px;
}
.ctrl-card h3::before {
  content:''; display:block; width:3px; height:13px;
  background:var(--accent); border-radius:2px;
}

/* D-Pad */
.dpad {
  display:grid; grid-template-columns:repeat(3,52px);
  grid-template-rows:repeat(3,52px);
  gap:5px; margin:0 auto 12px; width:fit-content;
}
.dpad-btn {
  width:52px; height:52px; border:1px solid var(--border);
  border-radius:10px; background:var(--raised); color:var(--text);
  font-size:19px; cursor:pointer; display:flex; align-items:center;
  justify-content:center; transition:all .12s;
  user-select:none; -webkit-user-select:none; touch-action:none;
}
.dpad-btn:hover { background:var(--bg2); border-color:var(--accent); color:var(--accent); }
.dpad-btn.pressed {
  background:rgba(34,211,238,.15); border-color:var(--accent);
  color:var(--accent); transform:scale(.92);
}
.dpad-btn.stop-btn {
  background:rgba(248,113,113,.08); border-color:rgba(248,113,113,.25); color:var(--red);
}
.dpad-btn.stop-btn:hover { background:rgba(248,113,113,.18); }

.spin-row { display:flex; gap:7px; margin-bottom:12px; }
.spin-btn {
  flex:1; padding:9px 4px; border:1px solid var(--border); border-radius:9px;
  background:var(--raised); color:var(--text2); font-size:12px; font-weight:600;
  cursor:pointer; font-family:'Outfit',sans-serif; transition:all .12s;
  user-select:none; -webkit-user-select:none; touch-action:none;
  display:flex; align-items:center; justify-content:center; gap:5px;
}
.spin-btn:hover { border-color:var(--accent); color:var(--accent); background:var(--glow); }
.spin-btn.pressed {
  background:rgba(34,211,238,.15); border-color:var(--accent);
  color:var(--accent); transform:scale(.95);
}

/* Sliders */
input[type=range] {
  -webkit-appearance:none; appearance:none;
  flex:1; height:5px; border-radius:3px;
  background:var(--raised); outline:none; cursor:pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance:none; appearance:none;
  width:15px; height:15px; border-radius:50%;
  background:var(--accent); cursor:pointer; box-shadow:0 0 7px var(--glow);
}
input[type=range]::-moz-range-thumb {
  width:15px; height:15px; border-radius:50%;
  background:var(--accent); cursor:pointer; border:none;
}

.speed-row { display:flex; align-items:center; gap:9px; }
.speed-row label { font-size:11px; color:var(--text2); white-space:nowrap; }
.val-badge {
  font-family:'JetBrains Mono',monospace; font-size:11px;
  color:var(--accent); min-width:30px; text-align:right;
}
.kb-hint {
  margin-top:9px; font-size:10.5px; color:var(--muted);
  text-align:center; font-family:'JetBrains Mono',monospace;
}

/* Servo */
.servo-row { display:flex; align-items:center; gap:9px; margin-bottom:9px; }
.servo-label { font-size:11px; color:var(--text2); min-width:54px; }

/* Tracking */
.track-section { margin-top:13px; padding-top:13px; border-top:1px solid var(--border); }
.track-section h3 { margin-bottom:11px; }
.btn-track {
  width:100%; padding:11px; border:none; border-radius:9px;
  font-family:'Outfit',sans-serif; font-size:13px; font-weight:700;
  cursor:pointer; transition:all .2s; letter-spacing:.2px;
}
.btn-track.off {
  background:linear-gradient(135deg,var(--green),#059669);
  color:#fff; box-shadow:0 4px 14px rgba(52,211,153,.3);
}
.btn-track.on {
  background:linear-gradient(135deg,var(--red),#dc2626);
  color:#fff; box-shadow:0 4px 14px rgba(248,113,113,.3);
}
.btn-track:hover { transform:translateY(-1px); filter:brightness(1.1); }

/* Sensor indicators */
.sensor-row {
  display:flex; gap:7px; margin-top:10px;
  justify-content:center; align-items:center;
}
.sensor-item { display:flex; flex-direction:column; align-items:center; gap:4px; }
.sensor-led {
  width:18px; height:18px; border-radius:50%;
  background:var(--raised); border:1px solid var(--border);
  transition:all .2s;
}
.sensor-led.active {
  background:var(--green); border-color:var(--green);
  box-shadow:0 0 8px rgba(52,211,153,.6);
}
.sensor-lbl { font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--muted); }

/* ── Detection panel ─────────────────────────────────────────────────────── */
.det-panel {
  background:var(--card); border-radius:13px;
  border:1px solid var(--border); padding:15px;
}
.det-panel h3 {
  font-size:12px; font-weight:700; color:var(--text2); text-transform:uppercase;
  letter-spacing:.5px; margin-bottom:13px;
  display:flex; align-items:center; gap:7px;
}
.det-panel h3::before {
  content:''; display:block; width:3px; height:13px;
  background:var(--amber); border-radius:2px;
}
.det-cam-row { display:flex; gap:8px; margin-bottom:13px; }
.btn-det {
  flex:1; padding:9px 6px; border-radius:9px; border:none;
  font-family:'Outfit',sans-serif; font-size:12px; font-weight:700;
  cursor:pointer; transition:all .2s;
}
.btn-det.off {
  background:rgba(100,116,139,.15); color:var(--muted);
  border:1px solid var(--border);
}
.btn-det.off:hover { color:var(--text2); border-color:var(--text2); }
.btn-det.on {
  background:linear-gradient(135deg,var(--amber),#d97706);
  color:#000; box-shadow:0 3px 12px rgba(251,191,36,.3);
}
.det-conf-row { display:flex; align-items:center; gap:9px; margin-bottom:13px; }
.det-conf-row label { font-size:11px; color:var(--text2); min-width:78px; }
.det-stats {
  display:flex; align-items:center; gap:12px;
  font-size:11px; color:var(--text2); margin-bottom:10px;
  font-family:'JetBrains Mono',monospace;
}
.det-count { color:var(--amber); font-weight:700; font-size:14px; }
.det-tags {
  display:flex; flex-wrap:wrap; gap:5px;
  min-height:22px; max-height:70px; overflow-y:auto;
}
.det-tag {
  padding:2px 9px; border-radius:4px; font-size:10px; font-weight:600;
  background:rgba(251,191,36,.12); color:var(--amber);
  font-family:'JetBrains Mono',monospace; border:1px solid rgba(251,191,36,.2);
}
.det-unavail {
  font-size:11px; color:var(--red); text-align:center; padding:8px 0;
}

/* ── Settings tab ────────────────────────────────────────────────────────── */
.settings-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; align-items:start; }
.settings-card {
  background:var(--card); border-radius:13px;
  border:1px solid var(--border); padding:15px;
}
.settings-card h3 {
  font-size:12px; font-weight:700; color:var(--text2); text-transform:uppercase;
  letter-spacing:.5px; margin-bottom:13px;
  display:flex; align-items:center; gap:7px;
}
.settings-card h3::before {
  content:''; display:block; width:3px; height:13px;
  background:var(--pink); border-radius:2px;
}
.setting-row { display:flex; align-items:center; gap:9px; margin-bottom:11px; }
.setting-row:last-child { margin-bottom:0; }
.setting-label { font-size:11px; color:var(--text2); min-width:120px; }

.save-bar { display:flex; align-items:center; gap:12px; margin-top:12px; }
.btn-save {
  padding:10px 22px; background:linear-gradient(135deg,var(--pink),#db2777);
  color:#fff; border:none; border-radius:9px;
  font-family:'Outfit',sans-serif; font-size:13px; font-weight:700;
  cursor:pointer; transition:all .2s;
}
.btn-save:hover { transform:translateY(-1px); filter:brightness(1.1); }
.save-ok {
  font-size:12px; color:var(--green);
  opacity:0; transition:opacity .3s;
}
.save-ok.show { opacity:1; }
.btn-reset {
  padding:10px 16px; background:transparent;
  color:var(--muted); border:1px solid var(--border);
  border-radius:9px; font-family:'Outfit',sans-serif; font-size:12px;
  font-weight:600; cursor:pointer; transition:all .2s;
}
.btn-reset:hover { color:var(--text2); border-color:var(--text2); }

/* ── Log panel ───────────────────────────────────────────────────────────── */
.log-panel {
  background:var(--card); border-radius:13px;
  border:1px solid var(--border); display:flex;
  flex-direction:column; overflow:hidden;
}
.log-hdr {
  display:flex; align-items:center; justify-content:space-between;
  padding:11px 15px; background:var(--bg2);
  border-bottom:1px solid var(--border); flex-shrink:0;
}
.log-hdr h2 { font-size:13px; font-weight:600; display:flex; align-items:center; gap:7px; }
.log-count {
  font-family:'JetBrains Mono',monospace; font-size:10px;
  color:var(--muted); background:var(--bg); padding:2px 7px; border-radius:5px;
}
.log-clear {
  background:none; border:1px solid var(--border); color:var(--muted);
  padding:3px 9px; border-radius:5px; font-size:11px;
  cursor:pointer; font-family:'Outfit',sans-serif;
}
.log-clear:hover { color:var(--text); border-color:var(--text2); }
.log-body {
  flex:1; overflow-y:auto; padding:5px 0;
  font-family:'JetBrains Mono',monospace; font-size:11.5px; line-height:1.65;
}
.log-body::-webkit-scrollbar { width:5px; }
.log-body::-webkit-scrollbar-thumb { background:var(--raised); border-radius:3px; }
.log-entry { padding:2px 15px; display:flex; gap:8px; }
.log-entry:hover { background:rgba(255,255,255,.02); }
.log-time { color:var(--muted); min-width:82px; }
.log-lvl {
  min-width:40px; font-size:9.5px; font-weight:700;
  padding:1px 5px; border-radius:3px; text-align:center;
  align-self:flex-start; margin-top:2px;
}
.log-lvl.INFO  { background:rgba(34,211,238,.12);  color:var(--accent); }
.log-lvl.ERROR { background:rgba(248,113,113,.12); color:var(--red); }
.log-lvl.WARN  { background:rgba(251,191,36,.12);  color:var(--amber); }
.log-msg { color:var(--text2); word-break:break-word; }

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width:960px) {
  body { overflow:auto; }
  .main { grid-template-columns:1fr; }
  .cam-grid, .ctrl-grid, .settings-grid { grid-template-columns:1fr; }
  .log-panel { max-height:260px; }
  .det-cam-row { flex-direction:column; }
}

/* Records tab */
.rec-card {
  background:var(--card); border-radius:10px; border:1px solid var(--border);
  overflow:hidden; display:flex; flex-direction:column;
}
.rec-card img {
  width:100%; aspect-ratio:4/3; object-fit:cover; background:#0d1420;
}
.rec-info {
  padding:8px 10px; font-size:11px;
}
.rec-label { color:var(--amber); font-weight:700; font-size:12px; }
.rec-meta { color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:10px; margin-top:3px; }

/* Toggle switch */
.toggle-wrap { display:flex; align-items:center; gap:8px; }
.toggle-switch {
  position:relative; width:36px; height:20px; background:var(--raised);
  border-radius:10px; cursor:pointer; transition:background .2s;
  border:1px solid var(--border);
}
.toggle-switch.on { background:var(--green); border-color:var(--green); }
.toggle-switch::after {
  content:''; position:absolute; top:2px; left:2px;
  width:14px; height:14px; border-radius:50%; background:#fff;
  transition:left .2s;
}
.toggle-switch.on::after { left:18px; }

/* Obstacle alarm blink */
@keyframes alarm-blink {
  0%,100%{background:rgba(248,113,113,.1);border-color:rgba(248,113,113,.3);}
  50%{background:rgba(248,113,113,.4);border-color:var(--red);}
}
.alarm-active {
  animation:alarm-blink 0.5s ease-in-out infinite;
  color:var(--red) !important;
}

</style>
</head>
<body>
<div class="app">

  <!-- ── Header ────────────────────────────────────────────────────────── -->
  <div class="header">
    <div class="logo">
      <div class="logo-icon">🗑️</div>
      <h1>Garbage <span>Detection</span> Dashboard <span style="font-size:11px;color:var(--amber);background:rgba(251,191,36,.12);padding:2px 7px;border-radius:4px;margin-left:4px">v2</span></h1>
    </div>

    <nav class="nav">
      <button class="nav-btn active" onclick="showTab('dashboard',this)">📹 Dashboard</button>
      <button class="nav-btn" onclick="showTab('controls',this)">🕹️ Controls</button>
      <button class="nav-btn" onclick="showTab('records',this)">📋 Records</button>
      <button class="nav-btn" onclick="showTab('settings',this)">⚙️ Settings</button>
    </nav>

    <div class="hdr-stats">
      <div>CAM0 <span class="sv off" id="h-c0">OFF</span></div>
      <div>CAM1 <span class="sv off" id="h-c1">OFF</span></div>
      <div>AI <span class="sv off" id="h-det">OFF</span></div>
      <div>Robot <span class="sv" id="h-robot">IDLE</span></div>
    </div>
  </div>

  <!-- ── Main ──────────────────────────────────────────────────────────── -->
  <div class="main">
    <div class="content-area">

      <!-- ══ Tab: Dashboard ════════════════════════════════════════════════ -->
      <div id="tab-dashboard" class="tab active">
        <div class="cam-grid">

          <!-- Camera 0 -->
          <div class="cam-card">
            <div class="cam-card-header">
              <div class="cam-title">
                <div class="cam-dot" id="dot0"></div>
                Camera 0
              </div>
              <span class="cam-fps" id="fps0">-- FPS</span>
              <button class="btn-cam-toggle is-open" id="btn-cam0" onclick="toggleCam(0)">
                Close
              </button>
            </div>
            <div class="cam-view">
              <img id="stream0" src="/video_feed/0" alt="Camera 0 stream">
            </div>
            <div class="cam-footer">
              <span id="frames0">0 frames</span>
              <span class="det-badge" id="det-badge0">● DETECTING</span>
              <button class="btn-cam-toggle" style="margin-left:auto;background:rgba(34,211,238,0.1);color:var(--accent);border-color:rgba(34,211,238,0.2)" onclick="takeSnapshot(0)">📸 Snapshot</button>
            </div>
          </div>

          <!-- Camera 1 -->
          <div class="cam-card">
            <div class="cam-card-header">
              <div class="cam-title">
                <div class="cam-dot" id="dot1"></div>
                Camera 1
              </div>
              <span class="cam-fps" id="fps1">-- FPS</span>
              <button class="btn-cam-toggle is-open" id="btn-cam1" onclick="toggleCam(1)">
                Close
              </button>
            </div>
            <div class="cam-view">
              <img id="stream1" src="/video_feed/1" alt="Camera 1 stream">
            </div>
            <div class="cam-footer">
              <span id="frames1">0 frames</span>
              <span class="det-badge" id="det-badge1">● DETECTING</span>
              <button class="btn-cam-toggle" style="margin-left:auto;background:rgba(34,211,238,0.1);color:var(--accent);border-color:rgba(34,211,238,0.2)" onclick="takeSnapshot(1)">📸 Snapshot</button>
            </div>
          </div>

        </div>

        <!-- Servo controls -->
        <div class="ctrl-card" style="margin-top:0">
          <h3>Servo Control</h3>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div class="servo-row">
              <span class="servo-label">Servo 1</span>
              <input type="range" min="0" max="180" value="90" id="db-s1-slider"
                oninput="onServo(1,this.value,'db-s1-val')">
              <span class="val-badge" id="db-s1-val">90°</span>
            </div>
            <div class="servo-row">
              <span class="servo-label">Servo 2</span>
              <input type="range" min="0" max="180" value="90" id="db-s2-slider"
                oninput="onServo(2,this.value,'db-s2-val')">
              <span class="val-badge" id="db-s2-val">90°</span>
            </div>
          </div>
        </div>

        <!-- ── Garbage Detection Panel ──────────────────────────────────── -->
        <div class="det-panel">
          <h3>
            Garbage Detection
            <span id="det-model-status" style="font-size:10px;text-transform:none;
              color:var(--muted);font-weight:400;margin-left:4px">(loading…)</span>
          </h3>

          <!-- Not available notice (hidden when YOLO is ready) -->
          <div class="det-unavail" id="det-unavail" style="display:none">
            ⚠ ultralytics not installed – run: <code>pip install ultralytics</code>
          </div>

          <div id="det-controls">
            <!-- Per-camera on/off -->
            <div class="det-cam-row">
              <button class="btn-det off" id="btn-det0" onclick="toggleDet(0)">
                📷 Cam 0 Detection: OFF
              </button>
              <button class="btn-det off" id="btn-det1" onclick="toggleDet(1)">
                📷 Cam 1 Detection: OFF
              </button>
            </div>

            <!-- Confidence threshold -->
            <div class="det-conf-row">
              <label>Confidence</label>
              <input type="range" min="10" max="95" value="45" id="det-conf-slider"
                oninput="setDetConf(this.value)">
              <span class="val-badge" id="det-conf-val">45%</span>
            </div>

            <!-- Detection stats -->
            <div class="det-stats">
              <span>Session total:</span>
              <span class="det-count" id="det-total">0</span>
              <span style="color:var(--border)">|</span>
              <span id="det-live-text" style="color:var(--muted)">No active detections</span>
            </div>

            <!-- Live detection tags -->
            <div class="det-tags" id="det-tags"></div>
          </div>
        </div>

      </div>
      <!-- ══ End Dashboard ════════════════════════════════════════════════ -->

      <!-- ══ Tab: Controls ════════════════════════════════════════════════ -->
      <div id="tab-controls" class="tab">
        <div class="ctrl-grid">

          <!-- Robot movement -->
          <div class="ctrl-card">
            <h3>Robot Movement</h3>
            <div class="dpad">
              <div></div>
              <button class="dpad-btn" id="btn-fwd"  data-dir="forward">↑</button>
              <div></div>
              <button class="dpad-btn" id="btn-left" data-dir="left">←</button>
              <button class="dpad-btn stop-btn" id="btn-stop">■</button>
              <button class="dpad-btn" id="btn-right" data-dir="right">→</button>
              <div></div>
              <button class="dpad-btn" id="btn-bwd"  data-dir="backward">↓</button>
              <div></div>
            </div>
            <div class="spin-row">
              <button class="spin-btn" id="btn-sl" data-dir="spin_left">↺ Spin L</button>
              <button class="spin-btn" id="btn-sr" data-dir="spin_right">↻ Spin R</button>
            </div>
            <div class="speed-row">
              <label>Speed</label>
              <input type="range" id="ctrl-speed" min="0" max="100" value="70"
                oninput="document.getElementById('ctrl-spd-val').textContent=this.value">
              <span class="val-badge" id="ctrl-spd-val">70</span>
            </div>
            <div class="kb-hint">⌨ WASD / Arrows · Space=Stop · Q/E=Spin</div>
          </div>

          <!-- Servo + Tracking -->
          <div class="ctrl-card">
            <h3>Servo Control</h3>
            <div class="servo-row">
              <span class="servo-label">Servo 1</span>
              <input type="range" min="0" max="180" value="90" id="s1-slider"
                oninput="onServo(1,this.value,'s1-val')">
              <span class="val-badge" id="s1-val">90°</span>
            </div>
            <div class="servo-row">
              <span class="servo-label">Servo 2</span>
              <input type="range" min="0" max="180" value="90" id="s2-slider"
                oninput="onServo(2,this.value,'s2-val')">
              <span class="val-badge" id="s2-val">90°</span>
            </div>

            <div class="track-section">
              <h3>Line Following</h3>
              <button class="btn-track off" id="btn-track" onclick="toggleTracking()">
                ▶ Start Line Following
              </button>
              <div class="sensor-row">
                <div class="sensor-item">
                  <div class="sensor-led" id="led-L1"></div>
                  <span class="sensor-lbl">L1</span>
                </div>
                <div class="sensor-item">
                  <div class="sensor-led" id="led-L2"></div>
                  <span class="sensor-lbl">L2</span>
                </div>
                <div style="width:12px"></div>
                <div class="sensor-item">
                  <div class="sensor-led" id="led-R1"></div>
                  <span class="sensor-lbl">R1</span>
                </div>
                <div class="sensor-item">
                  <div class="sensor-led" id="led-R2"></div>
                  <span class="sensor-lbl">R2</span>
                </div>
              </div>
              <div style="text-align:center;margin-top:6px;font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--muted)">
                GPIO · L1=13 L2=15 R1=11 R2=7
              </div>
            </div>
          </div>

        </div>
      </div>
      <!-- ══ End Controls ════════════════════════════════════════════════ -->

      <!-- ══ Tab: Settings ════════════════════════════════════════════════ -->
      <div id="tab-settings" class="tab">
        <div class="settings-grid">

          <div class="settings-card">
            <h3>Robot Speed</h3>
            <div class="setting-row">
              <span class="setting-label">Default Speed</span>
              <input type="range" min="0" max="100" id="p-robot-speed"
                oninput="sv('sv-robot-speed',this.value)">
              <span class="val-badge" id="sv-robot-speed">70</span>
            </div>

            <h3 style="margin-top:14px">Line Following Speeds</h3>
            <div class="setting-row">
              <span class="setting-label">Forward</span>
              <input type="range" min="0" max="100" id="p-track-forward"
                oninput="sv('sv-track-forward',this.value)">
              <span class="val-badge" id="sv-track-forward">70</span>
            </div>
            <div class="setting-row">
              <span class="setting-label">Right Turn</span>
              <input type="range" min="0" max="100" id="p-track-right"
                oninput="sv('sv-track-right',this.value)">
              <span class="val-badge" id="sv-track-right">60</span>
            </div>
            <div class="setting-row">
              <span class="setting-label">Left Turn</span>
              <input type="range" min="0" max="100" id="p-track-left"
                oninput="sv('sv-track-left',this.value)">
              <span class="val-badge" id="sv-track-left">60</span>
            </div>
          </div>

          <div class="settings-card">
            <h3>Servo Defaults</h3>
            <div class="setting-row">
              <span class="setting-label">Servo 1 Default</span>
              <input type="range" min="0" max="180" id="p-servo1"
                oninput="sv('sv-servo1',this.value,'°')">
              <span class="val-badge" id="sv-servo1">90°</span>
            </div>
            <div class="setting-row">
              <span class="setting-label">Servo 2 Default</span>
              <input type="range" min="0" max="180" id="p-servo2"
                oninput="sv('sv-servo2',this.value,'°')">
              <span class="val-badge" id="sv-servo2">90°</span>
            </div>

            <h3 style="margin-top:14px">Camera</h3>
            <div class="setting-row">
              <span class="setting-label">JPEG Quality</span>
              <input type="range" min="10" max="100" id="p-cam-quality"
                oninput="sv('sv-cam-quality',this.value,'%')">
              <span class="val-badge" id="sv-cam-quality">80%</span>
            </div>
          </div>

          <!-- Detection Class Settings -->
          <div class="settings-card" style="grid-column:span 2">
            <h3>🎯 Detection Classes</h3>
            <p style="font-size:11px;color:var(--muted);margin-bottom:12px">Enable/disable each class and set per-class confidence threshold</p>
            <div id="class-settings-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px"></div>
          </div>

          <!-- Robot Behavior -->
          <div class="settings-card" style="grid-column:span 2">
            <h3>🤖 Robot Behavior</h3>
            <div style="display:flex;gap:12px;flex-wrap:wrap">
              <button class="btn-det off" id="btn-stop-detect" onclick="toggleStopOnDetect()" style="flex:1;min-width:200px">
                🛑 Stop on Detection: OFF
              </button>
              <button class="btn-det off" id="btn-obstacle" onclick="toggleObstacle()" style="flex:1;min-width:200px">
                🔊 Obstacle Avoidance: OFF
              </button>
            </div>
            <div id="obstacle-info" style="margin-top:10px;font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace"></div>
          </div>

        </div>

        <div class="save-bar">
          <button class="btn-save" onclick="saveSettings()">💾 Save Settings</button>
          <button class="btn-reset" onclick="resetSettings()">↺ Reset Defaults</button>
          <span class="save-ok" id="save-ok">✓ Saved successfully</span>
        </div>
      </div>
      <!-- ══ End Settings ════════════════════════════════════════════════ -->

      <!-- ══ Tab: Records ═════════════════════════════════════════════════ -->
      <div id="tab-records" class="tab">
        <div class="det-panel">
          <h3>📋 Detection Records <span id="rec-count" style="font-size:10px;color:var(--muted);font-weight:400;margin-left:6px">0 records</span></h3>
          <div style="display:flex;gap:8px;margin-bottom:12px">
            <button class="btn-det off" onclick="clearRecords()" style="flex:0">🗑️ Clear All</button>
          </div>
          <div id="records-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;max-height:calc(100vh - 250px);overflow-y:auto"></div>
        </div>
      </div>
      <!-- ══ End Records ══════════════════════════════════════════════════ -->

    </div><!-- .content-area -->

    <!-- ── Log panel ──────────────────────────────────────────────────── -->
    <div class="log-panel">
      <div class="log-hdr">
        <h2>📋 System Log <span class="log-count" id="log-count">0</span></h2>
        <button class="log-clear" onclick="clearLogs()">Clear</button>
      </div>
      <div class="log-body" id="log-body"></div>
    </div>

  </div><!-- .main -->
</div><!-- .app -->

<script>
// ── State ─────────────────────────────────────────────────────────────────────
const camOpen  = [true, true];
const detOn    = [false, false];
let tracking   = false;
let detAvail   = false;

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

// ── Camera toggle ─────────────────────────────────────────────────────────────
async function toggleCam(id) {
  const action = camOpen[id] ? 'close' : 'open';
  try {
    const res  = await fetch(`/camera/${id}/${action}`, { method:'POST' });
    const data = await res.json();
    if (data.success) {
      camOpen[id] = !camOpen[id];
      updateCamUI(id);
      if (camOpen[id]) {
        document.getElementById('stream' + id).src =
          '/video_feed/' + id + '?t=' + Date.now();
      }
    }
  } catch(e) { console.error(e); }
}

function updateCamUI(id) {
  const open = camOpen[id];
  document.getElementById('dot' + id).className = 'cam-dot' + (open ? ' on' : '');
  const btn = document.getElementById('btn-cam' + id);
  if (open) {
    btn.textContent = 'Close'; btn.className = 'btn-cam-toggle is-open';
  } else {
    btn.textContent = 'Open';    btn.className   = 'btn-cam-toggle is-closed';
  }
}

// ── Snapshot ──────────────────────────────────────────────────────────────────
async function takeSnapshot(id) {
  try {
    const res = await fetch(`/camera/${id}/snapshot`, { method:'POST' });
    const data = await res.json();
    if (data.success) {
      alert('Snapshot saved to: ' + data.path);
    } else {
      alert('Snapshot failed: ' + data.error);
    }
  } catch(e) { console.error(e); }
}

// ── Detection toggle ───────────────────────────────────────────────────────────
async function toggleDet(camId) {
  if (!detAvail) return;
  detOn[camId] = !detOn[camId];
  const key = 'detect_cam' + camId;
  await fetch('/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ [key]: detOn[camId] ? 1 : 0 })
  });
  updateDetBtn(camId);
  updateHdrDet();
}

function updateDetBtn(camId) {
  const btn = document.getElementById('btn-det' + camId);
  if (detOn[camId]) {
    btn.textContent = '📷 Cam ' + camId + ' Detection: ON';
    btn.className   = 'btn-det on';
  } else {
    btn.textContent = '📷 Cam ' + camId + ' Detection: OFF';
    btn.className   = 'btn-det off';
  }
  // Badge on camera card footer
  const badge = document.getElementById('det-badge' + camId);
  if (badge) badge.className = 'det-badge' + (detOn[camId] ? ' active' : '');
}

function updateHdrDet() {
  const el  = document.getElementById('h-det');
  const any = detOn[0] || detOn[1];
  el.textContent = any ? 'ON' : 'OFF';
  el.className   = 'sv' + (any ? ' detecting' : ' off');
}

// Confidence threshold (auto-saves with debounce)
let _confTimer = null;
function setDetConf(val) {
  document.getElementById('det-conf-val').textContent = val + '%';
  document.getElementById('p-detect-confidence').value = val;
  document.getElementById('sv-detect-confidence').textContent = val + '%';
  clearTimeout(_confTimer);
  _confTimer = setTimeout(() => {
    fetch('/settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ detect_confidence: parseInt(val) })
    });
  }, 400);
}

// ── D-Pad: movement ───────────────────────────────────────────────────────────
function getSpeed() { return parseInt(document.getElementById('ctrl-speed').value); }

async function robotMove(dir) {
  setPressedUI(dir);
  await fetch('/robot/move', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ direction: dir, speed: getSpeed() })
  });
}

async function robotStop() {
  clearPressedUI();
  await fetch('/robot/stop', { method:'POST' });
}

function setPressedUI(dir) {
  clearPressedUI();
  const map = {
    forward:'btn-fwd', backward:'btn-bwd',
    left:'btn-left',   right:'btn-right',
    spin_left:'btn-sl', spin_right:'btn-sr'
  };
  const el = document.getElementById(map[dir]);
  if (el) el.classList.add('pressed');
}
function clearPressedUI() {
  document.querySelectorAll('.dpad-btn,.spin-btn').forEach(b => b.classList.remove('pressed'));
}

// Wire up D-Pad buttons
(function() {
  const dirBtns = document.querySelectorAll('.dpad-btn[data-dir],.spin-btn[data-dir]');
  dirBtns.forEach(btn => {
    const dir = btn.dataset.dir;
    btn.addEventListener('mousedown',   () => robotMove(dir));
    btn.addEventListener('mouseup',     () => robotStop());
    btn.addEventListener('mouseleave',  () => robotStop());
    btn.addEventListener('touchstart',  e => { e.preventDefault(); robotMove(dir); }, { passive:false });
    btn.addEventListener('touchend',    e => { e.preventDefault(); robotStop(); },    { passive:false });
    btn.addEventListener('touchcancel', e => { e.preventDefault(); robotStop(); },    { passive:false });
  });
  document.getElementById('btn-stop').addEventListener('click', robotStop);
})();

// ── Keyboard control ──────────────────────────────────────────────────────────
const KEY_DIR = {
  ArrowUp:'forward',   w:'forward',   W:'forward',
  ArrowDown:'backward',s:'backward',  S:'backward',
  ArrowLeft:'left',    a:'left',      A:'left',
  ArrowRight:'right',  d:'right',     D:'right',
  q:'spin_left',       Q:'spin_left',
  e:'spin_right',      E:'spin_right',
};
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT') return;
  if (ev.repeat) return;
  const dir = KEY_DIR[ev.key];
  if (dir) { ev.preventDefault(); robotMove(dir); }
  else if (ev.key === ' ') { ev.preventDefault(); robotStop(); }
});
document.addEventListener('keyup', ev => {
  if (KEY_DIR[ev.key]) robotStop();
});

// ── Servo control ─────────────────────────────────────────────────────────────
let _servoTimer = null;
function onServo(id, angle, labelId) {
  document.getElementById(labelId).textContent = angle + '°';
  clearTimeout(_servoTimer);
  _servoTimer = setTimeout(() => {
    fetch('/robot/servo', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ id, angle: parseInt(angle) })
    });
  }, 60);
}

// ── Line following toggle ─────────────────────────────────────────────────────
async function toggleTracking() {
  tracking = !tracking;
  await fetch('/robot/tracking', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ enabled: tracking })
  });
  syncTrackBtn();
}

function syncTrackBtn() {
  const btn = document.getElementById('btn-track');
  btn.textContent = tracking ? '⏹ Stop Line Following' : '▶ Start Line Following';
  btn.className   = 'btn-track ' + (tracking ? 'on' : 'off');
}

// ── Settings helpers ──────────────────────────────────────────────────────────
function sv(id, val, suffix='') {
  document.getElementById(id).textContent = val + suffix;
}

function setSlider(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = val;
  el.dispatchEvent(new Event('input'));
}

async function loadSettings() {
  try {
    const data = await (await fetch('/settings')).json();
    setSlider('p-robot-speed',   data.robot_speed);
    setSlider('p-track-forward', data.track_forward);
    setSlider('p-track-right',   data.track_right);
    setSlider('p-track-left',    data.track_left);
    setSlider('p-servo1',        data.servo1_angle);
    setSlider('p-servo2',        data.servo2_angle);
    setSlider('p-cam-quality',   data.cam_quality);
    setSlider('ctrl-speed',      data.robot_speed);
    const syncServo = (base, angle) => {
      setSlider(base + '-slider', angle);
      document.getElementById(base + '-val').textContent = angle + '°';
    };
    syncServo('s1',    data.servo1_angle);
    syncServo('s2',    data.servo2_angle);
    syncServo('db-s1', data.servo1_angle);
    syncServo('db-s2', data.servo2_angle);
    // Detection settings
    const conf = data.detect_confidence || 45;
    document.getElementById('det-conf-slider').value = conf;
    document.getElementById('det-conf-val').textContent = conf + '%';
    detOn[0] = !!data.detect_cam0;
    detOn[1] = !!data.detect_cam1;
    updateDetBtn(0);
    updateDetBtn(1);
    updateHdrDet();
  } catch(e) {}
}

async function saveSettings() {
  const g = id => parseInt(document.getElementById(id).value);
  await fetch('/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      robot_speed:   g('p-robot-speed'),
      track_forward: g('p-track-forward'),
      track_right:   g('p-track-right'),
      track_left:    g('p-track-left'),
      servo1_angle:  g('p-servo1'),
      servo2_angle:  g('p-servo2'),
      cam_quality:   g('p-cam-quality'),
      detect_confidence: g('det-conf-slider'),
    })
  });
  const ok = document.getElementById('save-ok');
  ok.classList.add('show');
  setTimeout(() => ok.classList.remove('show'), 2500);
}

async function resetSettings() {
  await fetch('/settings/reset', { method:'POST' });
  await loadSettings();
  const ok = document.getElementById('save-ok');
  ok.textContent = '✓ Reset to defaults';
  ok.classList.add('show');
  setTimeout(() => { ok.classList.remove('show'); ok.textContent = '✓ Saved successfully'; }, 2500);
}

// ── Status polling ────────────────────────────────────────────────────────────
async function poll() {
  try {
    const data = await (await fetch('/status')).json();

    // Header camera stats
    for (let i = 0; i < 2; i++) {
      const ci = data.cameras[i];
      camOpen[i] = ci.open;
      updateCamUI(i);
      const hEl = document.getElementById('h-c' + i);
      hEl.textContent = ci.open ? ci.fps + 'fps' : 'OFF';
      hEl.className   = 'sv' + (ci.open ? '' : ' off');
      document.getElementById('fps'    + i).textContent = ci.open ? ci.fps + ' FPS' : '-- FPS';
      document.getElementById('frames' + i).textContent = ci.frames.toLocaleString() + ' frames';
    }

    // Robot state in header
    const rEl = document.getElementById('h-robot');
    if (data.tracking) {
      rEl.textContent = 'TRACKING'; rEl.className = 'sv tracking';
    } else {
      rEl.textContent = 'IDLE'; rEl.className = 'sv';
    }
    tracking = data.tracking;
    syncTrackBtn();

    // Sensor LEDs
    if (data.sensors) {
      for (const [key, high] of Object.entries(data.sensors)) {
        const led = document.getElementById('led-' + key);
        if (led) led.className = 'sensor-led' + (!high ? ' active' : '');
      }
    }

    // Detection status
    if (data.detection) {
      const det = data.detection;
      detAvail = det.available;

      // Model availability UI
      const modelStatus = document.getElementById('det-model-status');
      const unavailDiv  = document.getElementById('det-unavail');
      const controls    = document.getElementById('det-controls');
      if (det.available) {
        modelStatus.textContent = '(YOLOv8n ready)';
        modelStatus.style.color = 'var(--green)';
        unavailDiv.style.display = 'none';
        controls.style.display   = '';
      } else {
        modelStatus.textContent = '';
        unavailDiv.style.display = '';
        controls.style.display   = 'none';
      }

      // Total count
      document.getElementById('det-total').textContent = det.total;

      // Live tags from both cameras
      const allDets = [...(det.cam0_dets || []), ...(det.cam1_dets || [])];
      const tagsEl  = document.getElementById('det-tags');
      const liveEl  = document.getElementById('det-live-text');
      if (allDets.length > 0) {
        liveEl.textContent = allDets.length + ' item(s) in view';
        liveEl.style.color = 'var(--amber)';
        tagsEl.innerHTML   = allDets.map(d =>
          `<span class="det-tag">${d.label} ${d.conf}%</span>`
        ).join('');
      } else {
        liveEl.textContent = 'No items in view';
        liveEl.style.color = 'var(--muted)';
        tagsEl.innerHTML   = '';
      }

      // Stop active indicator
      if (det.stop_active) {
        document.getElementById('h-robot').textContent = 'PAUSED';
        document.getElementById('h-robot').className = 'sv detecting';
      }
    }

    // Class settings sync
    if (data.class_settings) {
      classSettings = data.class_settings;
    }

    // Obstacle alarm
    if (data.obstacle) {
      const obs = data.obstacle;
      const infoEl = document.getElementById('obstacle-info');
      const btn = document.getElementById('btn-obstacle');
      if (obs.alarm && btn) {
        btn.classList.add('alarm-active');
        if (infoEl) infoEl.textContent = '⚠️ OBSTACLE DETECTED! Distance: ' + obs.distance + 'cm';
      } else if (btn) {
        btn.classList.remove('alarm-active');
        if (infoEl) infoEl.textContent = obs.running ? 'Distance: ' + (obs.distance > 0 ? obs.distance + 'cm' : '--') : '';
      }
    }

    // Sync toggle states
    stopOnDetect = !!data.stop_on_detect;
    obstacleAvoid = !!data.obstacle_avoid;
    syncStopDetectBtn();
    syncObstacleBtn();

    // Logs
    const body = document.getElementById('log-body');
    const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
    body.innerHTML = '';
    data.logs.forEach(entry => {
      const d = document.createElement('div');
      d.className = 'log-entry';
      d.innerHTML =
        `<span class="log-time">${entry.time}</span>` +
        `<span class="log-lvl ${entry.level}">${entry.level}</span>` +
        `<span class="log-msg">${entry.msg}</span>`;
      body.appendChild(d);
    });
    document.getElementById('log-count').textContent = data.logs.length;
    if (atBottom) body.scrollTop = body.scrollHeight;

  } catch(e) {}
}

async function clearLogs() {
  await fetch('/clear_logs', { method:'POST' });
}

// ── Class settings rendering ──────────────────────────────────────────────────
const ALL_CLS = ['battery','can','cardboard','drink carton','glass bottle',
                 'paper','plastic bag','plastic bottle','plastic bottle cap','pop tab'];
let classSettings = {};

function renderClassSettings() {
  const grid = document.getElementById('class-settings-grid');
  if (!grid) return;
  grid.innerHTML = ALL_CLS.map(cls => {
    const key = cls.replace(/ /g,'_');
    const en = classSettings[cls]?.enabled ?? 1;
    const conf = classSettings[cls]?.conf ?? 70;
    return `<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--bg2);border-radius:8px;border:1px solid var(--border)">
      <div class="toggle-switch ${en?'on':''}" onclick="toggleClass('${cls}')" id="cls-tog-${key}"></div>
      <span style="flex:1;font-size:11px;color:${en?'var(--text)':'var(--muted)'}">${cls}</span>
      <input type="range" min="10" max="95" value="${conf}" style="width:60px"
        oninput="setClassConf('${cls}',this.value)" ${en?'':'disabled'}>
      <span class="val-badge" id="cls-conf-${key}" style="min-width:28px">${conf}%</span>
    </div>`;
  }).join('');
}

function toggleClass(cls) {
  const key = cls.replace(/ /g,'_');
  const cur = classSettings[cls]?.enabled ?? 1;
  const nv = cur ? 0 : 1;
  fetch('/settings', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ [`cls_en_${key}`]: nv })
  });
  if (!classSettings[cls]) classSettings[cls] = {};
  classSettings[cls].enabled = nv;
  renderClassSettings();
}

function setClassConf(cls, val) {
  const key = cls.replace(/ /g,'_');
  document.getElementById('cls-conf-'+key).textContent = val+'%';
  fetch('/settings', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ [`cls_conf_${key}`]: parseInt(val) })
  });
  if (!classSettings[cls]) classSettings[cls] = {};
  classSettings[cls].conf = parseInt(val);
}

// ── Stop on detect / Obstacle avoidance toggles ──────────────────────────────
let stopOnDetect = false;
let obstacleAvoid = false;

async function toggleStopOnDetect() {
  stopOnDetect = !stopOnDetect;
  await fetch('/stop_on_detect/toggle', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ enabled: stopOnDetect })
  });
  syncStopDetectBtn();
}
function syncStopDetectBtn() {
  const btn = document.getElementById('btn-stop-detect');
  btn.textContent = stopOnDetect ? '🛑 Stop on Detection: ON' : '🛑 Stop on Detection: OFF';
  btn.className = 'btn-det ' + (stopOnDetect ? 'on' : 'off');
}

async function toggleObstacle() {
  obstacleAvoid = !obstacleAvoid;
  await fetch('/obstacle/toggle', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ enabled: obstacleAvoid })
  });
  syncObstacleBtn();
}
function syncObstacleBtn() {
  const btn = document.getElementById('btn-obstacle');
  btn.textContent = obstacleAvoid ? '🔊 Obstacle Avoidance: ON' : '🔊 Obstacle Avoidance: OFF';
  btn.className = 'btn-det ' + (obstacleAvoid ? 'on' : 'off');
}

// ── Records ───────────────────────────────────────────────────────────────────
let lastRecCount = 0;
async function pollRecords() {
  try {
    const data = await (await fetch('/records')).json();
    const grid = document.getElementById('records-grid');
    const countEl = document.getElementById('rec-count');
    if (!grid) return;
    countEl.textContent = data.records.length + ' records';
    if (data.records.length === lastRecCount) return;
    lastRecCount = data.records.length;
    grid.innerHTML = data.records.reverse().map(r =>
      `<div class="rec-card">
        <img src="/records/image/${r.file}" alt="${r.label}" loading="lazy">
        <div class="rec-info">
          <div class="rec-label">#${r.id} ${r.label}</div>
          <div class="rec-meta">${r.conf}% · ${r.time}</div>
        </div>
      </div>`
    ).join('');
  } catch(e) {}
}

async function clearRecords() {
  await fetch('/records/clear', { method:'POST' });
  lastRecCount = 0;
  pollRecords();
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadSettings();
renderClassSettings();
setInterval(poll, 1000);
setInterval(pollRecords, 3000);
poll();
pollRecords();
</script>
</body>
</html>
"""


# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    if cam_id not in (0, 1):
        return "Invalid camera ID", 400
    return Response(
        cameras[cam_id].stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/camera/<int:cam_id>/open', methods=['POST'])
def camera_open(cam_id):
    if cam_id not in (0, 1):
        return jsonify({"success": False, "error": "Invalid camera ID"}), 400
    ok = cameras[cam_id].open()
    return jsonify({"success": ok, "id": cam_id})


@app.route('/camera/<int:cam_id>/close', methods=['POST'])
def camera_close(cam_id):
    if cam_id not in (0, 1):
        return jsonify({"success": False, "error": "Invalid camera ID"}), 400
    cameras[cam_id].close()
    return jsonify({"success": True, "id": cam_id})


@app.route('/camera/<int:cam_id>/snapshot', methods=['POST'])
def camera_snapshot(cam_id):
    if cam_id not in (0, 1):
        return jsonify({"success": False, "error": "Invalid camera ID"}), 400
    
    frame = cameras[cam_id].frame
    if not frame:
        return jsonify({"success": False, "error": "No frame available"}), 400
    
    snap_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    
    filename = f"snap_cam{cam_id}_{int(time.time())}.jpg"
    filepath = os.path.join(snap_dir, filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(frame)
        log.log(f"Snapshot saved: {filename}")
        return jsonify({"success": True, "path": filename})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/status')
def status():
    return jsonify({
        "cameras":   [c.info for c in cameras],
        "tracking":  robot.tracking,
        "sensors":   robot.sensor_states,
        "detection": detector.status,
        "obstacle":  obstacle.info,
        "class_settings": {cls: {"enabled": settings.get(f"cls_en_{cls.replace(' ','_')}",1),
                                 "conf": settings.get(f"cls_conf_{cls.replace(' ','_')}",70)}
                           for cls in ALL_CLASSES},
        "stop_on_detect": settings.get("stop_on_detect", 0),
        "obstacle_avoid": settings.get("obstacle_avoid", 0),
        "logs":      log.entries,
    })


@app.route('/records')
def get_records():
    return jsonify({"records": detector._records[-100:]})


@app.route('/records/image/<path:filename>')
def record_image(filename):
    rec_dir = os.path.join(os.path.dirname(__file__), "records")
    from flask import send_from_directory
    return send_from_directory(rec_dir, filename)


@app.route('/records/clear', methods=['POST'])
def clear_records():
    detector._records.clear()
    detector._record_id = 0
    detector._seen_objects.clear()
    log.log("Detection records cleared")
    return jsonify({"success": True})


@app.route('/obstacle/toggle', methods=['POST'])
def toggle_obstacle():
    data = request.get_json(force=True)
    enabled = bool(data.get("enabled", False))
    settings["obstacle_avoid"] = 1 if enabled else 0
    if enabled:
        obstacle.start()
    else:
        obstacle.stop()
    return jsonify({"success": True, "enabled": enabled})


@app.route('/stop_on_detect/toggle', methods=['POST'])
def toggle_stop_on_detect():
    data = request.get_json(force=True)
    enabled = bool(data.get("enabled", False))
    settings["stop_on_detect"] = 1 if enabled else 0
    log.log(f"Stop-on-detect: {'ON' if enabled else 'OFF'}")
    return jsonify({"success": True, "enabled": enabled})


@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    log.clear()
    return jsonify({"success": True})


@app.route('/robot/move', methods=['POST'])
def robot_move():
    data      = request.get_json(force=True)
    direction = data.get("direction", "")
    speed     = data.get("speed")
    robot.move(direction, speed)
    return jsonify({"success": True})


@app.route('/robot/stop', methods=['POST'])
def robot_stop():
    robot.stop()
    return jsonify({"success": True})


@app.route('/robot/servo', methods=['POST'])
def robot_servo():
    data  = request.get_json(force=True)
    sid   = data.get("id", 1)
    angle = data.get("angle", 90)
    robot.servo(sid, angle)
    return jsonify({"success": True})


@app.route('/robot/tracking', methods=['POST'])
def robot_tracking():
    data    = request.get_json(force=True)
    enabled = bool(data.get("enabled", False))
    if enabled:
        robot.start_tracking()
    else:
        robot.stop_tracking()
    return jsonify({"success": True, "tracking": robot.tracking})


@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)


@app.route('/settings', methods=['POST'])
def post_settings():
    data = request.get_json(force=True)
    for key in list(DEFAULTS.keys()) + [k for k in data if k.startswith('cls_')]:
        if key in data:
            if key in DEFAULTS:
                settings[key] = type(DEFAULTS[key])(data[key])
            else:
                settings[key] = int(data[key])
    log.log("Settings updated")
    return jsonify({"success": True, "settings": settings})


@app.route('/settings/reset', methods=['POST'])
def reset_settings():
    settings.update(DEFAULTS)
    log.log("Settings reset to defaults")
    return jsonify({"success": True, "settings": settings})


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    try:
        for cam in cameras:
            cam.open()
        log.log("Garbage Detection Dashboard v2 starting on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cameras:
            cam.close()
        robot.cleanup()
        print("Shutdown complete.")

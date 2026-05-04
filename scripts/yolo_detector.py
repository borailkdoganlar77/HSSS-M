import threading
import logging
import time
import numpy as np
import cv2
from ultralytics import YOLO
from http.server import BaseHTTPRequestHandler, HTTPServer
import io

# YOLO logger ayarları
yolo_logger = logging.getLogger("yolo_detector")
yolo_logger.setLevel(logging.INFO)
yolo_logger.propagate = False
fh = logging.FileHandler("yolo_detections.log")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
yolo_logger.addHandler(fh)


class MjpegStreamServer:
    """YOLO annotated frame'leri HTTP MJPEG olarak yayınlar."""

    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._frame_bytes = None          # En son JPEG frame
        self._clients = []                # Aktif client yazıcıları
        self._clients_lock = threading.Lock()
        self._server = None
        self._thread = None

    def push_frame(self, bgr_frame):
        ok, buf = cv2.imencode(
            ".jpg", bgr_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )
        if not ok:
            return
        data = buf.tobytes()
        with self._lock:
            self._frame_bytes = data
        with self._clients_lock:
            dead = []
            for q in self._clients:
                try:
                    # Her zaman sadece en son frame kalsın
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except Exception:
                            break
                    q.put_nowait(data)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)

    def _make_handler(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass   # HTTP loglarını sustur

            def do_GET(self):
                if self.path != "/stream":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=--frame"
                )
                self.end_headers()

                import queue
                q = queue.Queue(maxsize=4)
                with server_ref._clients_lock:
                    server_ref._clients.append(q)

                # Bağlanan client'a hemen son frame'i gönder
                with server_ref._lock:
                    if server_ref._frame_bytes:
                        try:
                            q.put_nowait(server_ref._frame_bytes)
                        except Exception:
                            pass

                try:
                    while True:
                        try:
                            frame_bytes = q.get(timeout=5)
                        except Exception:
                            # Timeout — client hâlâ bağlı, tekrar dene
                            continue

                        try:
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + frame_bytes +
                                b"\r\n"
                            )
                            self.wfile.flush()
                        except Exception:
                            break
                finally:
                    with server_ref._clients_lock:
                        if q in server_ref._clients:
                            server_ref._clients.remove(q)

        return Handler

    def start(self):
        try:
            handler = self._make_handler()
            self._server = HTTPServer((self.host, self.port), handler)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True
            )
            self._thread.start()
            yolo_logger.info(f"MJPEG stream başlatıldı: http://{self.host}:{self.port}/stream")
            print(f"[MjpegStreamServer] Başlatıldı: http://{self.host}:{self.port}/stream")
        except Exception as e:
            print(f"[MjpegStreamServer] HATA - başlatılamadı: {e}")
            yolo_logger.error(f"MJPEG server başlatılamadı: {e}")

    def stop(self):
        if self._server:
            self._server.shutdown()


class YoloDetectorThread:
    def __init__(
        self,
        model_path: str,
        camera,
        conf_threshold=0.4,
        interval=0.033,
        window_name="YOLO Camera",
        stream_host="0.0.0.0",
        stream_port=8080,
        show_window=True,
    ):
        self.model = YOLO(model_path)
        self.camera = camera
        self.conf_threshold = conf_threshold
        self.interval = interval
        self.window_name = window_name
        self.show_window = show_window

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # MJPEG stream server
        self.stream_server = MjpegStreamServer(host=stream_host, port=stream_port)

    def start(self):
        self.stream_server.start()
        yolo_logger.info("YOLO detector thread started.")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        self.stream_server.stop()
        if self.show_window:
            cv2.destroyAllWindows()
        yolo_logger.info("YOLO detector thread stopped.")

    def _get_frame(self):
        try:
            rgba = self.camera.get_rgba()
            if rgba is None:
                return None
            if rgba.dtype != np.uint8:
                rgba = (rgba * 255).astype(np.uint8)
            return rgba[:, :, :3]
        except Exception as e:
            yolo_logger.warning(f"Frame alınamadı: {e}")
            return None

    def _run(self):
        while not self._stop_event.is_set():
            frame_rgb = self._get_frame()

            if frame_rgb is not None and frame_rgb.size > 0:
                results = self.model.predict(
                    frame_rgb,
                    conf=self.conf_threshold,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                # plot() BGR döndürür, imshow ve MJPEG için BGR kullan
                final_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                # Frame boyutunu küçült — bant genişliği azalır
                h, w = final_frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    final_frame = cv2.resize(
                        final_frame,
                        (640, int(h * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )

                # MJPEG stream'e gönder
                self.stream_server.push_frame(final_frame)

                # Lokal pencere (sunucuda display varsa)
                if self.show_window:
                    cv2.imshow(self.window_name, final_frame)
                    key = cv2.waitKey(1)
                    if key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                # Tespitleri logla
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id]
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()
                            yolo_logger.info(
                                f"DETECT | class={cls_name} | conf={conf:.2f} | "
                                f"bbox=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]"
                            )
                    else:
                        yolo_logger.info("No detection")

            time.sleep(self.interval)

        if self.show_window:
            cv2.destroyAllWindows()
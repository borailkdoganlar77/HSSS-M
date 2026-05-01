import threading
import logging
import time
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO logger ayarları
yolo_logger = logging.getLogger("yolo_detector")
yolo_logger.setLevel(logging.INFO)
yolo_logger.propagate = False
fh = logging.FileHandler("yolo_detections.log")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
yolo_logger.addHandler(fh)

class YoloDetectorThread:
    def __init__(self, model_path: str, camera, conf_threshold=0.4, interval=0.033,window_name="YOLO Camera"):
        self.model = YOLO(model_path)
        self.camera = camera
        self.conf_threshold = conf_threshold
        self.interval = interval
        self.window_name = window_name
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        yolo_logger.info("YOLO detector thread started.")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        cv2.destroyAllWindows()
        yolo_logger.info("YOLO detector thread stopped.")

    def _get_frame(self):
        """Camera'dan numpy RGB frame al ve normalize et."""
        try:
            rgba = self.camera.get_rgba()
            if rgba is None:
                return None
            
            # 1. Float32 (0.0-1.0) geliyorsa Uint8 (0-255) formatına çevir
            if rgba.dtype != np.uint8:
                rgba = (rgba * 255).astype(np.uint8)
            
            # 2. Alpha kanalını at ve RGB olarak döndür
            return rgba[:, :, :3]
        except Exception as e:
            yolo_logger.warning(f"Frame alınamadı: {e}")
            return None

    def _run(self):
        while not self._stop_event.is_set():
            frame_rgb = self._get_frame()

            if frame_rgb is not None and frame_rgb.size > 0:
                # YOLO tahmini (YOLOv8 içeride RGB-BGR dönüşümünü yönetebilir ancak 
                # imshow için biz manuel yöneteceğiz)
                results = self.model.predict(
                    frame_rgb,
                    conf=self.conf_threshold,
                    verbose=False
                )

                # 3. RENK DÜZELTME: Ultralytics plot() BGR döndürür. 
                # Eğer frame_rgb kullanıyorsak, imshow öncesi BGR'ye dönüştürmeliyiz.
                annotated_frame = results[0].plot()
                
                # Simülasyon RGB, OpenCV BGR beklediği için:
                final_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("YOLO Camera", final_frame)
                key = cv2.waitKey(1)

                if key == 27 or cv2.getWindowProperty("YOLO Camera", cv2.WND_PROP_VISIBLE) < 1:
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

        cv2.destroyAllWindows()
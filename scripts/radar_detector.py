import threading
import time
import numpy as np
import cv2

class RadarDisplayThread:
    def __init__(self, radar_path, interval=0.05, map_size=600, max_range=150.0):
        # SimulationApp başladıktan sonra çağrılacağı için burada hata vermez
        import omni.replicator.core as rep
        import omni.kit.app
        
        # Extension'ın aktif olduğundan emin oluyoruz
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        if not ext_manager.is_extension_enabled("omni.replicator.core"):
            ext_manager.set_extension_enabled_immediate("omni.replicator.core", True)

        self.radar_path = radar_path
        self.interval = interval
        self.map_size = map_size
        self.max_range = max_range
        
        # Annotator bağlantısı
        self.annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacComputeRTXRadarPointCloud")
        self.annotator.attach([self.radar_path])
        
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.center = map_size // 2
        self.scale = self.center / max_range

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        cv2.destroyWindow("Radar Scope")

    def _run(self):
        while not self._stop_event.is_set():
            data = self.annotator.get_data()
            frame = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
            cv2.circle(frame, (self.center, self.center), 5, (255, 0, 0), -1) # İstasyon
            
            if data is not None and "detections" in data:
                for det in data["detections"]:
                    px_x = int(self.center + (det['x'] * self.scale))
                    px_y = int(self.center - (det['y'] * self.scale))
                    if 0 <= px_x < self.map_size and 0 <= px_y < self.map_size:
                        cv2.circle(frame, (px_x, px_y), 4, (0, 0, 255), -1)

            cv2.imshow("Radar Scope", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            time.sleep(self.interval)
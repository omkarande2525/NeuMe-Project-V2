import time
import numpy as np
from shared.logger import get_logger

class RealTimeInferenceLoop:
    """System Engine: Simulates real-time inference loop."""
    def __init__(self, config, extractor, predictor, director):
        self.config = config
        self.extractor = extractor
        self.predictor = predictor
        self.director = director
        self.logger = get_logger("RealTimeLoop")
        
        self.eeg_stream = None
        self.fnirs_stream = None
        self._running = False
        
    def load_replay_data(self, eeg_epochs: list, fnirs_epochs: list):
        self.eeg_stream = eeg_epochs
        self.fnirs_stream = fnirs_epochs
        
    def run(self, replay_speed: float = 1.0, duration_s: float = 60.0):
        if not self.eeg_stream or not self.fnirs_stream:
            self.logger.error("No replay data loaded")
            return
            
        self._running = True
        step_s = self.config['inference']['step_size_s']
        max_idx = min(len(self.eeg_stream), len(self.fnirs_stream))
        
        idx = 0
        start_time = time.time()
        
        self.logger.info("Starting real-time loop...")
        
        while self._running and idx < max_idx:
            t0 = time.perf_counter()
            
            # Simulated elapsed check for duration limit
            if time.time() - start_time > duration_s:
                break
                
            try:
                # 1. Get window
                eeg_w = self.eeg_stream[idx]
                fnirs_w = self.fnirs_stream[idx]
                
                # 2. Extract Features
                z = self.extractor.extract(eeg_w, fnirs_w)
                
                # 3. Predict
                state, conf = self.predictor.predict(z)
                
                # 4. Game Update
                features_dict = {"signal_energy": z[3], "modality_ratio": z[12]}
                params = self.director.get_params(state, conf, features_dict)
                
            except Exception as e:
                self.logger.warning(f"Cycle {idx} failed: {e}. Attempting recovery.")
                state, conf, params = "UNKNOWN", 0.0, {}
                
            # 5. Timing
            t_end = time.perf_counter()
            latency_ms = (t_end - t0) * 1000
            
            # Print output strictly for demo
            print(f"[LIVE] STATE: {state:<7} (conf: {conf:.2f}) | Latency: {latency_ms:.1f}ms")
            
            # Sleep logic
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, (step_s / replay_speed) - elapsed)
            time.sleep(sleep_time)
            
            idx += 1
            
        self.logger.info("Real-time loop completed.")

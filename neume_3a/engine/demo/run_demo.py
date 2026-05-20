import argparse
import time
import yaml
from colorama import init, Fore, Style

from features.feature_extractor import FeatureExtractor
from classification.predictor import CognitiveStatePredictor
from engine.llm_bridge.game_director import GameDirector
from engine.inference.realtime_loop import RealTimeInferenceLoop
from features.data_loader import TensorDataLoader

init(autoreset=True)

def print_header():
    print(Fore.CYAN + "==========================================================")
    print(Fore.CYAN + "|  Neume 3A - Neural Inference Engine - Live Demo         |")
    print(Fore.CYAN + "==========================================================")

def run_demo(speed: float, duration: float):
    print_header()
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(Fore.YELLOW + "Loading System Components...")
    extractor = FeatureExtractor()
    extractor.load("data/processed/normalizer.pkl")
    
    predictor = CognitiveStatePredictor("rf")
    predictor.load("data/processed/rf_baseline.pkl")
    
    director = GameDirector(config)
    
    loader = TensorDataLoader(config)
    # Load Subject 4 (Validation Subject) for demo
    print(Fore.YELLOW + "Loading Validation Stream (Subject 4)...")
    eeg, fnirs = loader.load_subject(4)
    
    loop = RealTimeInferenceLoop(config, extractor, predictor, director)
    loop.load_replay_data(eeg, fnirs)
    
    print(Fore.GREEN + "System Ready. Starting Live Inference...")
    print("-" * 60)
    
    # Overwrite the loop's print logic temporarily to match the exact demo format requested
    # Actually, we can just run the loop if we modify the loop's print, or we monkeypatch it here
    
    start_time = time.time()
    for idx in range(min(len(eeg), len(fnirs))):
        if time.time() - start_time > duration:
            break
            
        t0 = time.perf_counter()
        
        # 1. Feature Extraction
        z = extractor.extract(eeg[idx], fnirs[idx])
        
        # 2. Predict
        state, conf = predictor.predict(z)
        
        # 3. Game Director
        features_dict = {"signal_energy": float(z[3]), "modality_ratio": float(z[12])}
        params = director.get_params(state, conf, features_dict)
        
        t_end = time.perf_counter()
        latency_ms = (t_end - t0) * 1000
        
        # Format output
        time_str = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
        color = Fore.GREEN if state == "FOCUSED" else Fore.CYAN if state == "FLOW" else Fore.RED
        
        print(f"[{time_str}] > {color}{state:<9}{Style.RESET_ALL} (conf: {conf:.2f}) | Latency: {latency_ms:.1f}ms")
        param_str = f"lanes={params['lanes']} speed={params['speed']:.1f} theme={params['environment_theme']} tone={params['narrative_tone']}"
        print(Fore.LIGHTBLACK_EX + f"        |- Game: {param_str}")
        
        # Sleep to simulate real-time
        elapsed = time.perf_counter() - t0
        sleep_time = max(0, (config['inference']['step_size_s'] / speed) - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    
    run_demo(args.speed, args.duration)

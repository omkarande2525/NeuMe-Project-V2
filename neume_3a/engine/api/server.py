from flask import Flask, jsonify, request
from flask_cors import CORS
import yaml
import numpy as np
from pathlib import Path

from shared.logger import get_logger
from features.feature_extractor import FeatureExtractor
from classification.predictor import CognitiveStatePredictor
from engine.llm_bridge.game_director import GameDirector

app = Flask(__name__)
CORS(app)

logger = get_logger("API_Server")

# Globals loaded ONCE at startup
_config = {}
_extractor = None
_predictor = None
_director = None
_latest_state = {"state": "UNKNOWN", "confidence": 0.0, "params": {}}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": _predictor._is_loaded})

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify(_latest_state)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        z_list = data.get("z_vector")
        if not z_list or len(z_list) != 15:
            return jsonify({"error": "Invalid z_vector"}), 400
            
        z = np.array(z_list, dtype=np.float32)
        
        state, conf = _predictor.predict(z)
        features_dict = {"signal_energy": float(z[3]), "modality_ratio": float(z[12])}
        params = _director.get_params(state, conf, features_dict)
        
        global _latest_state
        _latest_state = {
            "state": state,
            "confidence": conf,
            "params": params
        }
        
        return jsonify(_latest_state)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

def init_app():
    global _config, _extractor, _predictor, _director
    with open("config/config.yaml", "r") as f:
        _config = yaml.safe_load(f)
        
    _extractor = FeatureExtractor()
    if Path("data/processed/normalizer.pkl").exists():
        _extractor.load("data/processed/normalizer.pkl")
        
    _predictor = CognitiveStatePredictor("rf")
    if Path("data/processed/rf_baseline.pkl").exists():
        _predictor.load("data/processed/rf_baseline.pkl")
        
    _director = GameDirector(_config)
    logger.info("Server components initialized globally.")

if __name__ == '__main__':
    init_app()
    app.run(host="0.0.0.0", port=5050)

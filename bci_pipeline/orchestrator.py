# ============================================================
# FILE: bci_pipeline/orchestrator.py
# ============================================================
"""CLI orchestrator for the BCI data pipeline."""

import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

from loaders.physionet_loader import load_physionet
from loaders.seed_loader import load_seed
from loaders.fnirs_loader import load_fnirs
from pipelines.eeg_pipeline import run_eeg_pipeline
from pipelines.fnirs_pipeline import run_fnirs_pipeline
from validators import validate_eeg_epochs, validate_fnirs_array, validate_alignment
from utils.io_utils import save_pickle
from utils.logger import get_logger

log = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="BCI Pipeline Orchestrator")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)), help="List of subject IDs to process")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    parser.add_argument("--modality", type=str, choices=["physionet", "seed", "fnirs", "multimodal"], default="multimodal", 
                        help="Dataset/Modality to process. 'multimodal' implies PhysioNet + fNIRS.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip subjects already processed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        log.error(f"Failed to load config file {args.config}: {e}")
        return

    processed_dir = Path(config["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    summary = []
    
    for subject_id in tqdm(args.subjects, desc="Processing Subjects"):
        eeg_status, fnirs_status = "N/A", "N/A"
        eeg_shape, fnirs_shape = "N/A", "N/A"
        
        try:
            epochs = None
            fnirs_arr = None
            
            # 1. Processing EEG branch
            if args.modality in ["physionet", "seed", "multimodal"]:
                out_path_eeg = processed_dir / f"eeg_epochs_subj{subject_id:03d}.pkl"
                if args.skip_existing and out_path_eeg.exists():
                    log.info(f"Skipping existing EEG for subject {subject_id:03d}")
                    eeg_status = "Skipped"
                else:
                    if args.modality in ["physionet", "multimodal"]:
                        raw_eeg = load_physionet(subject_id, config)
                        log.info(f"Subject {subject_id:03d}: PhysioNet loader used")
                    else:
                        # seed only
                        raw_eeg = load_seed(subject_id, config)
                        log.info(f"Subject {subject_id:03d}: SEED-IV loader used")
                        
                    epochs = run_eeg_pipeline(raw_eeg, config)
                    validate_eeg_epochs(epochs, subject_id)
                    eeg_status = "Success"
                    eeg_shape = str(epochs.get_data().shape)

            # 2. Processing fNIRS branch
            if args.modality in ["fnirs", "multimodal"]:
                out_path_fnirs = processed_dir / f"fnirs_epochs_subj{subject_id:03d}.pkl"
                if args.skip_existing and out_path_fnirs.exists():
                    log.info(f"Skipping existing fNIRS for subject {subject_id:03d}")
                    fnirs_status = "Skipped"
                else:
                    raw_fnirs = load_fnirs(subject_id, config)
                    fnirs_arr = run_fnirs_pipeline(raw_fnirs, config)
                    validate_fnirs_array(fnirs_arr, subject_id)
                    fnirs_status = "Success"
                    fnirs_shape = str(fnirs_arr.shape)

            # 3. Alignment if Multimodal
            if args.modality == "multimodal" and epochs is not None and fnirs_arr is not None:
                epochs, fnirs_arr = validate_alignment(epochs, fnirs_arr, config, subject_id)
                eeg_shape = str(epochs.get_data().shape)
                fnirs_shape = str(fnirs_arr.shape)

            # 4. Save aligned files to disk
            if eeg_status == "Success":
                save_pickle(epochs, out_path_eeg)
            if fnirs_status == "Success":
                save_pickle(fnirs_arr, out_path_fnirs)
                
        except Exception as e:
            log.error(f"Subject {subject_id:03d} failed: {e}")
            if eeg_status not in ["Success", "Skipped"]: eeg_status = "Failed"
            if fnirs_status not in ["Success", "Skipped"]: fnirs_status = "Failed"
            
        summary.append((subject_id, eeg_status, fnirs_status, eeg_shape, fnirs_shape))

    print("\nProcessing Summary:")
    print("-" * 80)
    print(f"{'Subject':<10} | {'EEG Status':<12} | {'fNIRS Status':<14} | {'EEG Shape':<20} | {'fNIRS Shape':<20}")
    print("-" * 80)
    for row in summary:
        print(f"{row[0]:<10} | {row[1]:<12} | {row[2]:<14} | {row[3]:<20} | {row[4]:<20}")
    print("-" * 80)

if __name__ == "__main__":
    main()

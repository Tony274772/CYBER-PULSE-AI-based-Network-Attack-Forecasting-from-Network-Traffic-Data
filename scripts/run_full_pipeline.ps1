# Run Full Pipeline

# 1. Clean Flow Data
python data_prep/clean_cic2017.py --input_dir data/raw/cic2017/csv --output_dir data/processed/cic2017

# 2. Extract Packet Features
python data_prep/extract_packet_features.py --pcap data/raw/cic2017/pcap/Wednesday-workingHours.pcap --output_dir data/processed/cic2017 --delta_t_seconds 5

# 3. Build Temporal Dataset
python data_prep/build_temporal_dataset.py --clean_flow data/processed/cic2017/cic2017_clean.parquet --packet_features data/processed/cic2017/cic2017_wednesday_packet_features.parquet --mapping_file mitre_mapping/attack_stage_mapping.json --output_dir data/processed/cic2017

# 4. Train Baselines
python src/training/train_baselines.py

# 5. Train Temporal GRU
python src/training/train_world_model.py

# 6. Evaluate
python src/evaluation/run_eval.py

# 7. Start Dashboard
streamlit run dashboard/streamlit_app.py

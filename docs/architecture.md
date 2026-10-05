# CyberPulse Architecture (V2)

The system uses a Dual-Stream Temporal GRU World Model.

## Data Pipelines

1. **Flow Pipeline**: Parses CIC-IDS2017 or CTU-13 flow records into 5-second non-overlapping windows. Aggregates network state (28 features) at the window level.
2. **Packet Pipeline**: Parses raw PCAP streams into corresponding 5-second windows (12 features).

## Neural Architecture

- **Inputs**: 
  - Flow: 28-dim
  - Packet: 12-dim
- **Fusion**: Both streams are embedded into 32-dim vectors and concatenated into a 64-dim network state.
- **Temporal Component**: A GRUCell (hidden size 64) tracks historical network context over a 60-second window (12 steps of 5 seconds).
- **Outputs**:
  - `Next State Pred`: Predicts the network state for the next 5-second window.
  - `Risk Forecast`: Probability of attack at +5s, +30s, and +60s horizons.
  - `Stage Forecast`: Multiclass MITRE ATT&CK progression.

## Inference and Rollout

During inference, the model runs a recursive rollout: predicting the next state, feeding it back into the GRU, and outputting future forecast curves up to 60 seconds into the future.

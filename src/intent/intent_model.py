# ═══════════════════════════════════════════════════════
# FILE: src/intent/intent_model.py
# PURPOSE: Neural engine for recognizing operator intent from biological and state sequences.
# LAYER: Intent Layer
# INPUTS: 16-channel skeletal muscle (EMG) signals and 24-DOF body kinematics.
# OUTPUTS: Intent class logits, time-to-action predictions, and system confidence scores.
# CALLED BY: training/train_intent.py, src/robot/controllers/arm_pattern_recognizer.py (planned)
# ═══════════════════════════════════════════════════════

import time
import torch
import torch.nn as torch_nn
from typing import Dict, Tuple

class IntentTransformer(torch_nn.Module):
    """
    WHAT THIS CLASS IS:
      A deep learning brain based on the "Transformer" architecture that looks at 
      patterns in muscle signals and body movement over a short window of time 
      to guess what the human wants to do next.

    WHY IT EXISTS:
      The robot arms shouldn't just react to current signals; they need to "see" 
      the sequence of movement to differentiate between a casual reach and an 
      active grip attempt. This class solves the "anticipation" problem in ARM-S.

    HOW IT WORKS (step by step):
      1. Projects raw EMG and Body signals into a shared 'internal language' (64-dim).
      2. Adds both signals together to create a unified 'fused' feature set.
      3. Adds labels to each frame so the model knows the order (Positional Encoding).
      4. Passes the sequence through 3 layers of Transformer logic to find relationships.
      5. Looks at the very last frame in the window to make three final predictions.

    EXAMPLE USAGE:
      model = IntentTransformer(seq_len=25, emg_dim=16, body_dim=24)
      predictions = model(emg_batch, body_batch)
      intent_label = predictions['intent'].argmax(dim=-1)
    """
    def __init__(
        self, 
        seq_len: int = 25, 
        emg_dim: int = 16, 
        body_dim: int = 24, 
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 7
    ):
        """
        WHAT THIS DOES: Configures the neural network layers and hyperparameters.
        
        WHY: To establish the capacity and structure of the transformer before training starts.
        
        ARGS:
          seq_len (int): Number of history frames (usually 25 for a 500ms window at 50Hz).
          emg_dim (int): 16 channels of sensor data.
          body_dim (int): 24 degrees of freedom from the human model.
          d_model (int): Width of the internal representation (larger = smarter but slower).
          nhead (int): Number of 'attention' heads to look for different patterns simultaneously.
          num_layers (int): Depth of the brain's reasoning layers.
          dim_feedforward (int): Size of the reasoning 'muscle' inside each layer.
          dropout (float): Probability to ignore neurons to prevent the brain from memorizing patterns.
          num_classes (int): Total number of possible activities to recognize (e.g. REACH, GRASP).
        
        RETURNS:
          (None)
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Linear projection to lift low-level sensors to high-level features
        self.emg_proj = torch_nn.Linear(emg_dim, d_model)
        # Separate projection to ensure body kinematics aren't drowned out by EMG noise
        self.body_proj = torch_nn.Linear(body_dim, d_model)
        
        # Learnable positional encoding to tell the model 'when' a signal happened
        self.pos_encoder = torch_nn.Embedding(seq_len, d_model)
        
        # Main reasoning core using the Attention mechanism
        encoder_layer = torch_nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            # dropout helps generalization to new users
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = torch_nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # FINAL PREDICTION HEADS
        # Head 1: What is the user trying to do?
        self.intent_head = torch_nn.Linear(d_model, num_classes)
        # Head 2: How many seconds until the peak of the action?
        self.timing_head = torch_nn.Linear(d_model, 1)
        
        # Head 3: How sure are we of these predictions?
        self.confidence_proj = torch_nn.Linear(d_model, 1)
        # Sigmoid squashes output to [0, 1] to represent a percentage chance of success
        self.confidence_activation = torch_nn.Sigmoid()

    def forward(self, emg_seq: torch.Tensor, body_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        WHAT THIS DOES: Runs the actual calculation loop to turn sensors into a guess.
        
        WHY: This is the primary execution path for both training and live inference.
        
        ARGS:
          emg_seq (torch.Tensor): Shape (B, 25, 16) - raw muscle window.
          body_seq (torch.Tensor): Shape (B, 25, 24) - raw movement window.
            
        RETURNS:
          (Dict): Dictionary holding tensors for 'intent', 'timing', and 'confidence'.
        
        MATH/LOGIC:
          - Feature Fusion: Addition is chosen over Concatenation to keep state-space small (64-dim).
          - Sequential Reasoning: The Transformer looks back across the 25 frames to see if 
            movement is accelerating or decelerating as it relate to muscle spikes.
        """
        # Map raw inputs to internal feature space (64-dim)
        emg_feat = self.emg_proj(emg_seq)
        body_feat = self.body_proj(body_seq)
        
        # Fusion by addition allows the model to treat EMG and Body as complementary signals
        fused_feat = emg_feat + body_feat
        
        # Generate sequence IDs [0, 1, 2...seq_len-1] for the position brain
        positions = torch.arange(self.seq_len, device=fused_feat.device).unsqueeze(0)
        # Look up the 'meaning' of each position in time
        pos_feat = self.pos_encoder(positions)
        
        # Inject timing information into the fused features
        x = fused_feat + pos_feat
        
        # Run through the reasoning layers (The Transformer Encoder)
        encoded = self.transformer_encoder(x)
        
        # We only care about the very last state because transformers refine 
        # knowledge as the sequence progresses; index -1 contains the cumulative summary.
        cls_token = encoded[:, -1, :]  # Extract summary token (Batch, 64)
        
        # Split summarized intelligence into specialized outputs
        intent = self.intent_head(cls_token) # Direct classification
        timing = self.timing_head(cls_token) # Direct regression
        
        # Calculate system confidence
        conf_logits = self.confidence_proj(cls_token)
        # Map raw confidence to a probability score between 0 and 1
        confidence = self.confidence_activation(conf_logits)
        
        return {
            'intent': intent,
            'timing': timing,
            'confidence': confidence
        }

    def export_onnx(self, path: str) -> None:
        """
        WHAT THIS DOES: Saves the current brain to a file format other software can read easily.
        
        WHY: PyTorch is for learning; ONNX is for high-speed deployment on the robot's onboard chip.
        
        ARGS:
          path (str): where to save the file.
        
        RETURNS:
          (None)
        """
        # Ensure model is optimized for prediction, not learning (disables Dropout)
        self.eval()
        
        # Create fake data so the exporter can 'trace' the math path
        dummy_emg = torch.randn(1, self.seq_len, 16)
        dummy_body = torch.randn(1, self.seq_len, 24)
        
        print(f"Exporting ONNX model to {path}...")
        torch.onnx.export(
            self,
            (dummy_emg, dummy_body),
            path,
            export_params=True, # Include weights in the file
            opset_version=17,    # Modern ONNX standard for stability
            do_constant_folding=True, # Fold math constants to save energy
            input_names=['emg_seq', 'body_seq'],
            output_names=['intent', 'timing', 'confidence'],
            # Allow different users to have different window sizes if needed
            dynamic_axes={
                'emg_seq': {0: 'batch_size'},
                'body_seq': {0: 'batch_size'},
                'intent': {0: 'batch_size'},
                'timing': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        print("ONNX export completed.")


def inference_time_check():
    """
    WHAT THIS DOES: Runs 1000 'practice' guesses on the GPU and measures how long they take.
    
    WHY: In surgical robotics (ARM-S), high latency causes jittery movement. We must stay under 15ms.
    
    ARGS:
      (None)
    
    RETURNS:
      (None) - Raises error if too slow.
    
    MATH/LOGIC:
      - Uses GPU Events to bypass CPU bottlenecks for accurate nanosecond-level timing.
      - Mean latency is calculated as (Total Time / 1000) to smooth out fluctuations.
    """
    # Pick the fastest available hardware (NVIDIA GPU or CPU fallback)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IntentTransformer().to(device)
    # Switch to prediction mode
    model.eval()
    
    # Simulate a single user loop (Batch size 1)
    BATCH_SIZE = 1
    dummy_emg = torch.randn(BATCH_SIZE, 25, 16).to(device)
    dummy_body = torch.randn(BATCH_SIZE, 25, 24).to(device)
    
    # Priming the pump (Warn up the circuits)
    print("Warming up GPU...")
    for _ in range(100):
        with torch.no_grad(): # Use no_grad to save memory during check
            _ = model(dummy_emg, dummy_body)
    
    # CORE TIMING LOOP
    NUM_ITERS = 1000
    print(f"Running {NUM_ITERS} iterations on {device}...")
    
    # Initialize high-precision hardware timers
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Block until background tasks finish to get a clean start time
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(NUM_ITERS):
        with torch.no_grad():
            _ = model(dummy_emg, dummy_body)
    end_event.record()
    
    # Block until all practice guesses are done
    torch.cuda.synchronize()
    
    # Convert hardware cycles to real-world milliseconds
    total_time_ms = start_event.elapsed_time(end_event)
    mean_latency_ms = total_time_ms / NUM_ITERS
    
    print(f"Mean Inference Latency: {mean_latency_ms:.4f} ms")
    # Safety assertion to prevent deployment of sluggish models
    assert mean_latency_ms < 15.0, f"Latency {mean_latency_ms:.4f}ms exceeded 15ms SLA!"
    print("Fast Inference SLA MET (< 15ms).")

if __name__ == "__main__":
    # 1. Initialize the Brain
    model = IntentTransformer()
    
    # 2. Audit complexity (How many 'memory cells' are used)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"IntentTransformer Parameters: {num_params:,}")
    
    # 3. Ensure this design is fast enough for real-time control
    inference_time_check()
    
    # 4. Prepare for robot deployment
    import os
    os.makedirs('checkpoints', exist_ok=True)
    model.export_onnx('checkpoints/intent_model.onnx')

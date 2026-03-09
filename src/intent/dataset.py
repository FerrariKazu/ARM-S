# ═══════════════════════════════════════════════════════
# FILE: src/intent/dataset.py
# PURPOSE: The 'Schoolbook' for the AI; turns long recorded sessions into bite-sized lessons.
# LAYER: Intent Layer (Data Preparation)
# INPUTS: Massive log files (.npz) containing thousands of robot movements.
# OUTPUTS: Formatted 'Batches' of data that PyTorch can use to teach the neural brain.
# CALLED BY: training/train_intent.py (The training script)
# ═══════════════════════════════════════════════════════

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SRLDataset(Dataset):
    """
    WHAT THIS CLASS IS:
      A "Data Librarian". Its job is to organize the millions of numbers recorded 
      from the robot and serve them to the AI one 'flashcard' at a time.

    WHY IT EXISTS:
      AI models can't read raw files directly while training; they need the 
      data to be cleaned, resized, and randomized so they learn general 
      concepts instead of just memorizing specific files.

    HOW IT WORKS (step by step):
      1. Loads the .npz data into high-speed memory (RAM).
      2. Calculates 'Z-Scores' to ensure all numbers are in a similar range.
      3. For every frame requested, it looks back 500ms to build a 'Video Clip' of data.
      4. Applie 'Augmentation' (Random noise/shifts) to make the AI more robust.

    EXAMPLE USAGE:
      train_data = SRLDataset("path/to/logs.npz", augment=True)
      sample = train_data[0] # Get the first 'flashcard'
    """
    def __init__(self, npz_path: str, augment: bool = True):
        """
        WHAT THIS DOES: Boots up the dataset and calculates the 'Difficulty Curve'.
        
        WHY: To ensure 'Rare' actions (like Handoffs) are given more importance during training.
        
        ARGS:
          npz_path (str): source file.
          augment (bool): if true, it 'hallucinates' extra training examples for robustness.
        """
        super().__init__()
        self.augment = augment
        
        print(f"Loading data from {npz_path}...")
        data = np.load(npz_path)
        # Sequence data (N samples, 25 frames, channels)
        self.emg_data = data['emg']   
        self.body_data = data['state'] 
        # Target goals
        self.intent_data = data['labels'] 
        self.time_data = data['time_to_action']     
        
        # STANDARDIZATION MAP
        # Calculate the average and fluctuation (STD) of human movement
        self.body_mean = np.mean(self.body_data, axis=0, keepdims=True)
        # Add epsilon (1e-8) to prevent dividing by zero if a sensor never moved
        self.body_std = np.std(self.body_data, axis=0, keepdims=True) + 1e-8
        
        # IMPORTANCE WEIGHTING
        # If the user spends 90% of the time IDLE, the AI might get lazy.
        # We calculate how often each class appears to 'boost' the rare ones.
        unique_classes, counts = np.unique(self.intent_data, return_counts=True)
        freqs = np.zeros(7, dtype=np.float32)
        for cls, count in zip(unique_classes, counts):
            if cls < 7:
                freqs[int(cls)] = count
                
        # Inverse frequency math: Rare Action = Higher Reward for getting it right.
        inv_freqs = 1.0 / (freqs + 1e-6)
        # Normalize so the average weight is exactly 1.0
        self._class_weights = torch.FloatTensor(inv_freqs / np.sum(inv_freqs) * 7.0)

    @property
    def class_weights(self) -> torch.Tensor:
        """WHAT THIS DOES: Returns the 'Balance Sheet' for the AI's loss function."""
        return self._class_weights

    def __len__(self) -> int:
        """WHAT THIS DOES: Reports the total number of frames in the database."""
        return len(self.emg_data)

    def __getitem__(self, idx: int) -> dict:
        """
        WHAT THIS DOES: The main 'Extractor'. Gets one specific moment in time.
        
        WHY: To feed the AI a specific 'Question' (Sensors) and 'Answer' (Intent).
        
        ARGS:
          idx (int): the frame number to retrieve.
            
        RETURNS:
          flashcard (dict): A dictionary of Tensors (EMG, Body, Intent, Timing).
        """
        emg = self.emg_data[idx].copy()
        
        # THE SLIDING WINDOW (Temporal Context)
        # AI needs to see the *flow* of movement. We grab the current frame 
        # and the 24 frames BEFORE it to create a 500ms sequence.
        if idx >= 24:
            # Standard sequence
            body = self.body_data[idx-24:idx+1].copy()
        else:
            # START OF FILE PADDING: If we are at frame 2, we repeat frame 0 
            # to fill the 25-frame minimum required by the Transformer.
            pad_len = 24 - idx
            pad_vals = np.tile(self.body_data[0], (pad_len, 1))
            body = np.vstack([pad_vals, self.body_data[0:idx+1]]).copy()
            
        intent = self.intent_data[idx]
        t = self.time_data[idx]
        
        # --- DATA CLEANING ---
        # 1. EMG Scaling: Divide by 5.0 Volts to get a range of [-1.0, 1.0]
        emg = np.clip(emg / 5.0, -1.0, 1.0)
        
        # 2. Body State Standard Normalization (The Z-Score)
        # Subtract average and divide by variation to make the math 'stable' for learning
        body = (body - self.body_mean) / self.body_std
        
        # 3. Time Clipping: Clamp predictions to a realistic 0-5 second window
        t = np.clip(t, 0.0, 5.0)
        
        # --- ONLINE AUGMENTATIONS (Training Gym for the AI) ---
        if self.augment:
            # A. TEMPORAL SHIFT: Randomly wiggle the timeline by +/- 3 frames
            # This teaches the AI not to be too dependent on exact timing.
            if np.random.rand() < 0.5:
                shift = np.random.randint(-3, 4)
                if shift > 0:
                    emg = np.vstack([np.tile(emg[0], (shift, 1)), emg[:-shift]])
                    body = np.vstack([np.tile(body[0], (shift, 1)), body[:-shift]])
                elif shift < 0:
                    emg = np.vstack([emg[-shift:], np.tile(emg[-1], (-shift, 1))])
                    body = np.vstack([body[-shift:], np.tile(body[-1], (-shift, 1))])

            # B. SENSOR NOISE: Add random 'statick' to the EMG feed
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.02, emg.shape)
                emg = emg + noise
                
            # C. CHANNEL DROPOUT: Simulate a broken wire or loose sensor
            # This teaches the AI to focus on 'patterns' even if a muscle is missing.
            if np.random.rand() < 0.3:
                num_drop = np.random.randint(1, 3)
                channels_to_drop = np.random.choice(16, size=num_drop, replace=False)
                emg[:, channels_to_drop] = 0.0
                
        # Final safety clamp before converting to PyTorch tensors
        emg = np.clip(emg, -1.0, 1.0)
        
        return {
            'emg': torch.FloatTensor(emg),
            'body': torch.FloatTensor(body),
            'intent': torch.tensor(intent, dtype=torch.long),
            'time_to_action': torch.FloatTensor([t])
        }
        
    @staticmethod
    def collate_fn(batch):
        """
        WHAT THIS DOES: Stacks a group of 'flashcards' into a single 'Book' (Batch).
        
        WHY: Modern GPUs process data in parallel-groups for 10x faster training.
        """
        emg = torch.stack([x['emg'] for x in batch])
        body = torch.stack([x['body'] for x in batch])
        intent = torch.stack([x['intent'] for x in batch])
        time_to_action = torch.stack([x['time_to_action'] for x in batch])
        
        return {
            'emg': emg,
            'body': body,
            'intent': intent,
            'time_to_action': time_to_action
        }

if __name__ == "__main__":
    # SELF-TEST & VISUALIZATION SCRIPT
    data_path = "training/data/srl_dataset_train.npz"
    if not os.path.exists(data_path):
        print(f"File {data_path} not found. Please generate the dataset first.")
        exit(1)
        
    # Start the librarian in 'Learning Mode'
    dataset = SRLDataset(data_path, augment=True)
    print(f"\\nLoaded Dataset length: {len(dataset)}")
    
    # Verify our batching math works (Batch size 4)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=SRLDataset.collate_fn)
    batch = next(iter(loader))
    
    print("\\nBatch Shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} (dtype: {v.dtype})")
        
    # VALIDATION: Ensure the shapes match the IntentTransformer expectations
    assert batch['emg'].shape == (4, 25, 16), "Incorrect EMG shape"
    assert batch['body'].shape == (4, 25, 24), "Incorrect Body shape"
    
    # VISUAL AUDIT: Create a diagnostic plot to see what the AI 'sees'
    os.makedirs("notebooks", exist_ok=True)
    
    # Organize samples by task label for better comparison
    intent_samples = {}
    ds_intent = dataset.intent_data
    for target_class in sorted(np.unique(ds_intent)):
        if target_class >= 7: continue
        idx = np.where(ds_intent == target_class)[0][0]
        intent_samples[target_class] = dataset[idx]
        
    fig, axes = plt.subplots(len(intent_samples), 2, figsize=(10, 2.5 * len(intent_samples)))
    if len(intent_samples) == 1: axes = [axes] 
    
    class_names = ["IDLE", "REACH", "RETRACT", "GRASP", "RELEASE", "CARRY", "OVERHEAD"]
    
    for i, (cls_idx, sample) in enumerate(intent_samples.items()):
        emg = sample['emg'].numpy()
        body = sample['body'].numpy()
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"C_{cls_idx}"
        
        # PLOT 1: EMG HEATMAP (Visualizing muscle intensity over time)
        ax_emg = axes[i][0]
        im = ax_emg.imshow(emg.T, aspect='auto', cmap='plasma', vmin=-1.0, vmax=1.0)
        ax_emg.set_title(f"EMG [{cls_name}] (Normalized)")
        ax_emg.set_ylabel("Channel (1-16)")
        
        # PLOT 2: BODY KINEMATICS (Visualizing movement curves)
        ax_body = axes[i][1]
        for j in range(body.shape[1]):
            ax_body.plot(body[:, j], alpha=0.3)
        ax_body.set_title(f"Body [{cls_name}] (Standardized)")
        ax_body.set_ylabel("Z-Score")
        
        if i == len(intent_samples) - 1:
            ax_emg.set_xlabel("Time step")
            ax_body.set_xlabel("Time step")
            
    plt.tight_layout()
    output_path = "notebooks/dataset_sample.png"
    plt.savefig(output_path, dpi=120)
    print(f"\\nSaved diagnostic plot to {output_path}")

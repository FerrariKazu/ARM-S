# ═══════════════════════════════════════════════════════
# FILE: src/utils/profiler.py
# PURPOSE: The "Referee" for the robot's reaction time; ensures the math never runs too slow.
# LAYER: Monitoring Utility
# INPUTS: Names of software steps and their 'Time Budgets' (in milliseconds).
# OUTPUTS: A color-coded scoreboard showing which parts of the code are fast or slow.
# CALLED BY: src/sim/srl_env.py, training scripts, and performance audits.
# ═══════════════════════════════════════════════════════

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

@dataclass
class StageStats:
    """
    WHAT THIS IS:
      A simple 'Notebook' that records how many milliseconds a specific 
      job took over its last 100 attempts.
    """
    name: str # e.g. "Brain Calculation"
    budget_ms: float # The "Time Limit"
    # A sliding window that remembers the last 100 durations
    history: deque = field(default_factory=lambda: deque(maxlen=100))

class LatencyProfiler:
    """
    WHAT THIS CLASS IS:
      A digital 'Stopwatch'. It tracks how long every part of the robot's 
      thought process takes and warns the engineers if the robot is 
      starting to 'Lag'.

    WHY IT EXISTS:
      In robotics, 'Lag' is dangerous. If the 'Intent' model takes 0.5 seconds 
      longer than usual to think, the robot might hit something. This class 
      keeps the software 'Honest' and provides a performance report.

    HOW IT WORKS (step by step):
      1. You wrap a block of code with `with profiler("task_name"):`.
      2. The profiler marks the exact microsecond the code starts.
      3. It marks the microsecond it ends and calculates the difference.
      4. It averages these results over time to provide a stable 'Avg Response Time'.

    EXAMPLE USAGE:
      profiler = LatencyProfiler({"brain": 30.0})
      with profiler("brain"):
          run_ai_model()
      profiler.report()
    """
    
    # Terminal formatting codes for pretty-printing reports
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

    def __init__(self, budgets: Dict[str, float], window_size: int = 50):
        """
        WHAT THIS DOES: Sets the 'Speed Limits' for the entire system.
        """
        self.stages: Dict[str, StageStats] = {
            name: StageStats(name, budget, deque(maxlen=window_size))
            for name, budget in budgets.items()
        }
        self._current_stage: Optional[str] = None
        self._start_time: Optional[float] = None

    def __call__(self, stage_name: str):
        """
        WHAT THIS DOES: Selects which part of the code we are currently timing.
        """
        if stage_name not in self.stages:
            # Prevent silent failures if a developer typos a stage name
            raise ValueError(f"Unknown stage: {stage_name}")
        self._current_stage = stage_name
        return self

    def __enter__(self):
        """
        WHAT THIS DOES: The 'Click' of the stopwatch when code starts.
        """
        # High-resolution CPU counter (more accurate than time.time())
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        WHAT THIS DOES: The 'Stop' of the stopwatch when code finishes.
        """
        # Calculate duration and convert seconds to milliseconds
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        # Add to the historical average for this specific stage
        self.stages[self._current_stage].history.append(duration_ms)
        self._current_stage = None

    def rolling_average(self, stage_name: str) -> float:
        """WHAT THIS DOES: Calculates the 'Average Lag' over recent history."""
        history = self.stages[stage_name].history
        return sum(history) / len(history) if history else 0.0

    def report(self):
        """
        WHAT THIS DOES: Prints the "Final Scoreboard" to the terminal.
        
        WHY: To give developers an instant 'Pass/Fail' on system performance.
        """
        # PRINT HEADER
        print(f"\n{self.BOLD}{'Stage':<25} | {'Avg (ms)':<10} | {'Budget (ms)':<12} | {'Status'}{self.END}")
        print("-" * 65)
        
        total_avg = 0.0
        total_budget = 0.0
        
        for name, stats in self.stages.items():
            avg = self.rolling_average(name)
            budget = stats.budget_ms
            total_avg += avg
            total_budget += budget
            
            # --- EVALUATION LOGIC ---
            # If we are under budget, show GREEN. If we are lagging, show RED.
            color = self.GREEN if avg <= budget else self.RED
            status = f"{color}UNDER BUDGET{self.END}" if avg <= budget else f"{color}OVER BUDGET{self.END}"
            
            print(f"{name:<25} | {avg:<10.2f} | {budget:<12.2f} | {status}")
        
        print("-" * 65)
        # Final End-to-End Score
        total_color = self.GREEN if total_avg <= total_budget else self.RED
        print(f"{self.BOLD}{'TOTAL':<25} | {total_avg:<10.2f} | {total_budget:<12.2f} | {total_color}{'PASS' if total_avg <= total_budget else 'FAIL'}{self.END}")

if __name__ == "__main__":
    # PERFORMANCE STRESS TEST DEMO
    # These budgets match the hardware requirements of the physical robot
    mock_budgets = {
        "emg_acquisition": 5.0,
        "preprocessing": 5.0,
        "intent_prediction": 35.0,
        "context_classification": 10.0,
        "motion_planning": 30.0,
        "controller_execution": 5.0
    }
    
    profiler = LatencyProfiler(mock_budgets)
    
    print("Simulating 10 control loop iterations...")
    for _ in range(10):
        # 1. EMG Acquisition (FAST)
        with profiler("emg_acquisition"):
            time.sleep(0.004) # 4ms simulation
            
        # 2. Preprocessing (FAST)
        with profiler("preprocessing"):
            time.sleep(0.003) # 3ms simulation
            
        # 3. Intent Prediction (SLOW / OVER BUDGET)
        # Teaches developers what a failure looks like
        with profiler("intent_prediction"):
            time.sleep(0.038) # 38ms (budget is 35)
            
        # 4. Context Classification (FAST)
        with profiler("context_classification"):
            time.sleep(0.008) # 8ms
            
        # 5. Motion Planning (FAST)
        with profiler("motion_planning"):
            time.sleep(0.025) # 25ms
            
        # 6. Controller Execution (FAST)
        with profiler("controller_execution"):
            time.sleep(0.004) # 4ms
            
    # Print the colored diagnostics
    profiler.report()

#!/usr/bin/env python3
"""
TTT Prediction Broadcaster for Unity BCI Runner Game.

This module runs the TCFormer Hybrid model with TTT and broadcasts
predictions over WebSocket to connected Unity clients.

Usage:
    # Start the server with simulation mode (reads from GDF file):
    python -m intentflow.online.server.ttt_broadcaster \
        --checkpoint path/to/model.ckpt \
        --data_dir path/to/gdf \
        --subject 1 --session T \
        --port 8000

    # Unity connects to ws://localhost:8000/control
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import mne

# Ensure project paths are available
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "offline"))

from intentflow.online.models.online_wrapper import OnlineTCFormerWrapper
from intentflow.online.dsp.stream_preprocessor import WindowNormalizer


# -----------------------------------------------------------------------------
# Intent Message (compatible with app.py IntentMessage schema)
# -----------------------------------------------------------------------------

def create_intent_message(
    intent: str,
    confidence: float,
    timestamp: float,
) -> Dict[str, Any]:
    """Create an intent message compatible with Unity client."""
    return {
        "type": "intent",
        "intent": intent,
        "conf": confidence,
        "ts": timestamp,
        "protocol_version": 1,
    }


# -----------------------------------------------------------------------------
# Class Mapping
# -----------------------------------------------------------------------------

# Map BCIC2a class indices to control intents
# For 2-class control: left_hand (0) -> "left", right_hand (1) -> "right"
CLASS_TO_INTENT = {
    0: "left",    # left_hand
    1: "right",   # right_hand
    2: "idle",    # feet (ignored for lane control)
    3: "idle",    # tongue (ignored for lane control)
}


# -----------------------------------------------------------------------------
# Trial Extraction
# -----------------------------------------------------------------------------

def extract_trials_from_gdf(
    gdf_path: str,
    sfreq_target: int = 250,
    window_sec: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Extract motor imagery trials from a BCIC IV 2a GDF file."""
    event_codes = {'769': 0, '770': 1, '771': 2, '772': 3}
    
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    raw.pick(range(22))
    raw._data *= 1e6
    
    if raw.info['sfreq'] != sfreq_target:
        raw.resample(sfreq_target, verbose=False)
    
    sfreq = raw.info['sfreq']
    window_samples = int(window_sec * sfreq)
    
    trials = []
    labels = []
    onsets = []
    
    for annot in raw.annotations:
        if annot['description'] in event_codes:
            onset_sec = annot['onset']
            onset_sample = int(onset_sec * sfreq)
            end_sample = onset_sample + window_samples
            
            if end_sample > raw.n_times:
                continue
            
            trial_data = raw.get_data(start=onset_sample, stop=end_sample)
            trials.append(trial_data)
            labels.append(event_codes[annot['description']])
            onsets.append(onset_sec)
    
    if len(trials) == 0:
        return np.array([]), np.array([]), []
    
    return np.stack(trials, axis=0), np.array(labels), onsets


# -----------------------------------------------------------------------------
# WebSocket Server
# -----------------------------------------------------------------------------

class TTTBroadcaster:
    """
    Broadcasts TTT model predictions over WebSocket.
    
    This class manages:
    1. Model inference using OnlineTCFormerWrapper
    2. WebSocket connections to Unity clients
    3. Prediction rate control
    """
    
    def __init__(
        self,
        model_wrapper: OnlineTCFormerWrapper,
        normalizer: WindowNormalizer,
        prediction_interval: float = 0.5,
        confidence_threshold: float = 0.4,
        two_class_only: bool = True,
    ):
        """
        Initialize the broadcaster.
        
        Args:
            model_wrapper: The online model wrapper.
            normalizer: Data normalizer.
            prediction_interval: Minimum time between predictions (seconds).
            confidence_threshold: Minimum confidence to send a prediction.
            two_class_only: If True, only send left/right predictions.
        """
        self.model = model_wrapper
        self.normalizer = normalizer
        self.prediction_interval = prediction_interval
        self.confidence_threshold = confidence_threshold
        self.two_class_only = two_class_only
        
        self.clients: set = set()
        self.last_prediction_time = 0.0
        self.prediction_count = 0
        self.correct_count = 0
        
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        dead_clients = set()
        
        for client in list(self.clients):
            try:
                await client.send(message_str)
            except Exception as e:
                print(f"[Broadcast] Error sending to client: {e}")
                dead_clients.add(client)
        
        for client in dead_clients:
            self.clients.discard(client)
    
    async def process_trial(
        self,
        trial: np.ndarray,
        label: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single trial and broadcast the prediction.
        
        Args:
            trial: EEG data [C, T].
            label: Ground truth label (optional, for metrics).
            verbose: Print prediction info.
            
        Returns:
            Prediction result dictionary.
        """
        # Normalize
        normalized = self.normalizer(trial)
        
        # Predict
        result = self.model.predict_step(normalized)
        
        pred_idx = result['pred_idx']
        confidence = result['confidence']
        
        # Map to intent
        intent = CLASS_TO_INTENT.get(pred_idx, "idle")
        
        # For two-class mode, skip non-left/right predictions
        if self.two_class_only and intent == "idle":
            if verbose:
                print(f"  [Skip] pred={result['pred_label']}, conf={confidence:.2f} (not left/right)")
            return result
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            if verbose:
                print(f"  [Low Conf] pred={intent}, conf={confidence:.2f} < {self.confidence_threshold}")
            return result
        
        # Create and broadcast message
        current_time = time.time()
        message = create_intent_message(
            intent=intent,
            confidence=confidence,
            timestamp=current_time,
        )
        
        await self.broadcast(message)
        
        self.prediction_count += 1
        if label is not None and pred_idx == label:
            self.correct_count += 1
        
        if verbose:
            accuracy = (self.correct_count / self.prediction_count * 100) if self.prediction_count > 0 else 0
            gt_str = f", true={CLASS_TO_INTENT.get(label, 'unknown')}" if label is not None else ""
            print(f"  [Sent] {intent.upper()} (conf={confidence:.2f}{gt_str}) | "
                  f"Total: {self.prediction_count}, Acc: {accuracy:.1f}%")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "predictions": self.prediction_count,
            "correct": self.correct_count,
            "accuracy": (self.correct_count / self.prediction_count * 100) if self.prediction_count > 0 else 0,
            "clients": len(self.clients),
        }


# -----------------------------------------------------------------------------
# WebSocket Server using websockets library
# -----------------------------------------------------------------------------

async def run_websocket_server(broadcaster: TTTBroadcaster, host: str, port: int):
    """Run the WebSocket server."""
    try:
        import websockets
    except ImportError:
        print("Error: websockets library not installed. Run: pip install websockets")
        return
    
    async def handler(websocket):
        """Handle incoming WebSocket connections."""
        broadcaster.clients.add(websocket)
        print(f"[Server] Client connected ({len(broadcaster.clients)} total)")
        
        try:
            # Send welcome message
            welcome = {"type": "welcome", "message": "Connected to TTT Broadcaster"}
            await websocket.send(json.dumps(welcome))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            broadcaster.clients.discard(websocket)
            print(f"[Server] Client disconnected ({len(broadcaster.clients)} remaining)")
    
    print(f"[Server] Starting WebSocket server on ws://{host}:{port}")
    # origins=None disables origin checking for compatibility with Unity
    # ping_interval=None disables automatic ping to avoid connection issues
    async with websockets.serve(
        handler, 
        host, 
        port,
        origins=None,
        ping_interval=None,
        ping_timeout=None,
    ):
        await asyncio.Future()  # Run forever


async def run_simulation(
    broadcaster: TTTBroadcaster,
    trials: np.ndarray,
    labels: np.ndarray,
    trial_interval: float = 2.0,
    verbose: bool = True,
):
    """Run prediction simulation with trial data."""
    n_trials = len(trials)
    print(f"\n[Simulation] Starting with {n_trials} trials (interval: {trial_interval}s)")
    print(f"[Simulation] Waiting for clients to connect...")
    
    # Wait a bit for clients to connect
    await asyncio.sleep(2.0)
    
    for i, (trial, label) in enumerate(zip(trials, labels)):
        if verbose:
            print(f"\n[Trial {i+1}/{n_trials}]")
        
        await broadcaster.process_trial(trial, label=int(label), verbose=verbose)
        
        # Wait between trials
        await asyncio.sleep(trial_interval)
    
    stats = broadcaster.get_stats()
    print(f"\n[Simulation Complete]")
    print(f"  Total Predictions: {stats['predictions']}")
    print(f"  Accuracy: {stats['accuracy']:.1f}%")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTT Prediction Broadcaster for Unity BCI Runner"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (optional)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf",
        help="Path to BCIC IV 2a GDF files"
    )
    parser.add_argument(
        "--subject", type=int, default=1,
        help="Subject ID (1-9)"
    )
    parser.add_argument(
        "--session", type=str, default="T",
        choices=["T", "E"],
        help="Session: T (training) or E (evaluation)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="WebSocket server host"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="WebSocket server port"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU ID (-1 for CPU)"
    )
    parser.add_argument(
        "--trial_interval", type=float, default=2.0,
        help="Interval between trial predictions (seconds)"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.3,
        help="Minimum confidence to send prediction"
    )
    parser.add_argument(
        "--two_class_only", action="store_true", default=True,
        help="Only send left/right predictions"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop through trials continuously"
    )
    return parser.parse_args()


async def main_async(args):
    """Async main function."""
    device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu"
    
    print("=" * 60)
    print("TTT Prediction Broadcaster")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Subject: {args.subject}, Session: {args.session}")
    print(f"WebSocket: ws://{args.host}:{args.port}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\n[Init] Loading model...")
    model = OnlineTCFormerWrapper(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        reset_state_each_trial=False,
    )
    
    # Create normalizer
    normalizer = WindowNormalizer()
    
    # Create broadcaster
    broadcaster = TTTBroadcaster(
        model_wrapper=model,
        normalizer=normalizer,
        confidence_threshold=args.confidence_threshold,
        two_class_only=args.two_class_only,
    )
    
    # Load trial data
    gdf_path = Path(args.data_dir) / f"A{args.subject:02d}{args.session}.gdf"
    if not gdf_path.exists():
        print(f"Error: GDF file not found: {gdf_path}")
        return
    
    print(f"\n[Init] Loading data from: {gdf_path}")
    trials, labels, onsets = extract_trials_from_gdf(str(gdf_path))
    
    if len(trials) == 0:
        print("Error: No valid trials found")
        return
    
    # For two-class mode, filter to only left_hand (0) and right_hand (1)
    if args.two_class_only:
        mask = (labels == 0) | (labels == 1)
        trials = trials[mask]
        labels = labels[mask]
        print(f"[Init] Filtered to {len(trials)} left/right trials")
    
    print(f"[Init] Loaded {len(trials)} trials")
    print(f"[Init] Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Start server and simulation concurrently
    server_task = asyncio.create_task(
        run_websocket_server(broadcaster, args.host, args.port)
    )
    
    # Give server time to start
    await asyncio.sleep(1.0)
    
    # Run simulation
    if args.loop:
        print("\n[Mode] Loop mode enabled - will repeat trials continuously")
        while True:
            await run_simulation(
                broadcaster, trials, labels,
                trial_interval=args.trial_interval,
                verbose=True
            )
            model.reset_state()
            print("\n[Loop] Restarting simulation...")
            await asyncio.sleep(2.0)
    else:
        await run_simulation(
            broadcaster, trials, labels,
            trial_interval=args.trial_interval,
            verbose=True
        )
        
        # Keep server running after simulation
        print("\n[Server] Simulation complete. Server still running. Press Ctrl+C to exit.")
        await server_task


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")


if __name__ == "__main__":
    main()


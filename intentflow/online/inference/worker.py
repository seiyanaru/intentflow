"""Background worker that converts raw samples into intent messages."""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

import numpy as np

from intentflow.online.dsp.filters import StreamingFilter
from intentflow.online.inference.onnx_model import OnnxModel
from intentflow.online.inference.stabilizer import Stabilizer
from intentflow.online.server.app import INTENT_BUS, META, RAW_BUF, IntentMessage


async def inference_worker(
  onnx_path: str,
  stats_json: str,
  win_sec: float = 1.0,
  hop_sec: float = 0.2,
  conf_th: float = 0.7,
  ema_ms: int = 150,
  require_consec: int = 3,
  min_interval_ms: int = 200,
) -> None:
  """Continuously window buffered samples and emit stabilised intents."""
  stabilizer = Stabilizer(conf_th=conf_th, ema_ms=ema_ms, require_consec=require_consec, min_interval_ms=min_interval_ms)
  model: Optional[OnnxModel] = None
  filt: Optional[StreamingFilter] = None
  signature: Optional[Tuple[int, int]] = None
  pos = 0

  try:
    while True:
      try:
        fs = int(META.get("fs") or 0)
        ch = int(META.get("ch") or 0)
        if fs <= 0 or ch <= 0:
          await asyncio.sleep(0.01)
          continue

        need = max(1, int(win_sec * fs))
        hop = max(1, int(hop_sec * fs))

        if signature != (fs, ch) or model is None or filt is None:
          model = OnnxModel(onnx_path, stats_json)
          if model.C != ch:
            raise ValueError(f"Channel mismatch: stats expect {model.C}, ingest provides {ch}")
          filt = StreamingFilter(fs=fs, ch=ch)
          signature = (fs, ch)
          pos = max(0, len(RAW_BUF) - need)

        buffer_len = len(RAW_BUF)
        if buffer_len < need:
          pos = max(0, buffer_len - need)
          await asyncio.sleep(0.01)
          continue
        if buffer_len - pos < need:
          pos = max(0, buffer_len - need)
          await asyncio.sleep(0.01)
          continue

        snapshot = np.asarray(list(RAW_BUF), dtype=np.float32)
        buffer_len = snapshot.shape[0]

        while buffer_len - pos >= need:
          window = snapshot[pos : pos + need]
          pos += hop
          if window.shape[0] < model.T:
            break
          if window.shape[0] > model.T:
            window = window[-model.T :]
          filtered = filt.apply(window)
          features = filtered.T.astype(np.float32, copy=False)
          prepped = model.preprocess(features)
          logits, probs = model.infer(prepped)
          prob_vec = probs[0]
          pred_idx = int(np.argmax(prob_vec))
          intent_label = "left" if pred_idx == 0 else "right"
          conf = float(np.max(prob_vec))
          result = stabilizer.apply(intent_label, conf)
          stable_intent = result["intent"]
          if stable_intent != "none":
            message = IntentMessage(intent=stable_intent, conf=float(result["conf_ema"]), ts=time.time())
            await INTENT_BUS.put(message.model_dump_json())
        current_len = len(RAW_BUF)
        if pos > current_len:
          pos = max(0, current_len - hop)
        await asyncio.sleep(0.01)
      except asyncio.CancelledError:
        raise
      except Exception as exc:  # noqa: BLE001
        print(f"inference_worker error: {exc}")
        await asyncio.sleep(0.2)
  except asyncio.CancelledError:
    return

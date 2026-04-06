# Realtime Unicorn Setup

## Goal
- まずは `Unicorn -> UDP -> intentflow -> WebSocket -> Unity` の疎通を安定化する
- 現在の BCIC2a 22ch モデルは Unicorn 8ch に対して未校正なので、分類精度ではなく transport と推論安定性を先に確認する

## Current Constraint
- 学習済みモデルは BCIC2a 22ch 前提
- Unicorn の live input は 8ch
- 現在の online path はチャンネル数不一致を zero-pad / truncate で吸収している
- したがって、今の live 推論は `smoke test` として扱うべき

## Recommended Runtime
- `window_sec=4.0`
- `hop_sec=0.25`
- `stream_sfreq=250`
- `udp_packet_format=auto`
- `confidence_threshold=0.55`
- `normalizer=window` から開始

`normalizer=stream` は長時間安定性を見るときに試す。最初は `window` の方が挙動が読みやすい。
純正 `Unicorn UDP` ツールを使う場合、packet は CSV ではなく binary 17-float なので、`udp_packet_format=auto` または `unicorn17f` を使う。

## Launch on Lab PC
```bash
cd /workspace-cloud/seiya.narukawa/intentflow
./intentflow/online/scripts/run_unicorn_live.sh
```

subject を切り替える場合:
```bash
cd /workspace-cloud/seiya.narukawa/intentflow
SUBJECT_ID=7 ./intentflow/online/scripts/run_unicorn_live.sh
```

ポートを変える場合:
```bash
cd /workspace-cloud/seiya.narukawa/intentflow
WS_PORT=9001 UDP_PORT=11001 ./intentflow/online/scripts/run_unicorn_live.sh
```

## Success Criteria
- 研究室PCで `Unicorn UDP reader initialized` が出る
- UDP packet が流れ始める
- Unity 側が `ws://<lab-pc-ip>:8765` に接続できる
- `LEFT` / `RIGHT` intent が継続送信される
- 推論時間が実時間内に収まる

## Windows Bridge
Windows 側では `Unicorn Recorder` だけでは研究室PCに EEG が流れない。別ブリッジが必要。

最小構成:
```bash
python intentflow/online/scripts/unicorn_udp_bridge.py \
  --source lsl \
  --target-host 192.168.100.12 \
  --target-port 11001 \
  --channels 8 \
  --samples-per-packet 1 \
  --stream-type EEG \
  --verbose
```

LSL がまだ無い場合の疎通確認:
```bash
python intentflow/online/scripts/unicorn_udp_bridge.py \
  --source mock \
  --target-host 192.168.100.12 \
  --target-port 11001 \
  --channels 8 \
  --samples-per-packet 1 \
  --verbose
```

備考:
- `lsl` モードには `pylsl` が必要
- Unicorn 側で LSL 配信できない場合は、別途 SDK ベースの取得アプリが必要
- 研究室PC側は `UDP_PORT=11001 ./intentflow/online/scripts/run_unicorn_live.sh`

## Next Practical Step
- 今回は online 推論の本番評価ではなく、通信確認と calibration data の収集を優先する
- 5分程度で `rest / left / right` を記録し、Unicorn 8ch 専用の baseline を作る
- その後に online OTTA を議論する

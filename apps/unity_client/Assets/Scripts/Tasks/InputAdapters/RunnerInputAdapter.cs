using System;
using System.IO;
using UnityEngine;
using IntentFlow.Inputs;
using Tasks.Runner3Lane.Core;

namespace Tasks.Runner3Lane.InputAdapters
{
    public class RunnerInputAdapter : MonoBehaviour
    {
        // #region Debug Helper
        private static void DebugLog(string hyp, string loc, string msg, object data = null)
        {
            try
            {
                var json = $"{{\"hypothesisId\":\"{hyp}\",\"location\":\"{loc}\",\"message\":\"{msg}\",\"data\":\"{data}\",\"timestamp\":{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}}}";
                File.AppendAllText("/workspace-cloud/seiya.narukawa/intentflow/.cursor/debug.log", json + "\n");
            }
            catch { }
        }
        // #endregion
        [SerializeField] private RunnerController runner;
        [SerializeField] private MonoBehaviour sourceBehaviour;
        [SerializeField] private bool allowKeyboardInput = false;  // Disabled by default for BCI mode

        public int SuccessCount { get; private set; }
        public int FalseMoveCount { get; private set; }
        
        /// <summary>Last applied signal with full latency info.</summary>
        public IntentSignal? LastAppliedSignal { get; private set; }

        public event Action<IntentType> ActionTaken;
        
        /// <summary>Event fired when a BCI signal is applied, includes full timing info.</summary>
        public event Action<IntentSignal> SignalApplied;

        private IIntentSource _source;
        private bool _enabled = true;

        public bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        private void Awake()
        {
            CacheSource();
            if (!runner)
            {
                runner = FindObjectOfType<RunnerController>();
            }
            // #region agent log
            DebugLog("D", "RunnerInputAdapter:Awake", "init_state", $"source={_source != null},runner={runner != null},sourceBehaviour={sourceBehaviour?.name}");
            // #endregion
        }

        public void SetSource(IIntentSource source)
        {
            _source = source;
            sourceBehaviour = source as MonoBehaviour;
        }

        public void SetSource(MonoBehaviour behaviour)
        {
            sourceBehaviour = behaviour;
            CacheSource();
        }

        public void ResetStats()
        {
            SuccessCount = 0;
            FalseMoveCount = 0;
        }

        private void Update()
        {
            if (!_enabled || runner == null)
            {
                return;
            }

            // Keyboard input (for testing)
            if (allowKeyboardInput)
            {
                if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A))
                {
                    runner.MoveLeft();
                    SuccessCount++;
                    ActionTaken?.Invoke(IntentType.Left);
                    return;
                }
                if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D))
                {
                    runner.MoveRight();
                    SuccessCount++;
                    ActionTaken?.Invoke(IntentType.Right);
                    return;
                }
            }

            // BCI input from MiSource
            if (_source != null && _source.TryRead(out var signal))
            {
                // Record apply timestamp
                signal.ApplyTs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                
                // #region agent log
                DebugLog("D", "RunnerInputAdapter:Update", "bci_signal_received", $"type={signal.Type},conf={signal.Confidence}");
                // #endregion
                
                switch (signal.Type)
                {
                    case IntentType.Left:
                        runner.MoveLeft();
                        SuccessCount++;
                        ActionTaken?.Invoke(IntentType.Left);
                        break;
                    case IntentType.Right:
                        runner.MoveRight();
                        SuccessCount++;
                        ActionTaken?.Invoke(IntentType.Right);
                        break;
                    default:
                        FalseMoveCount++;
                        ActionTaken?.Invoke(IntentType.Idle);
                        break;
                }
                
                // Store and fire event with full timing info
                LastAppliedSignal = signal;
                SignalApplied?.Invoke(signal);
                
                // Log latency info
                if (signal.TotalLatency >= 0)
                {
                    string correctStr = signal.IsCorrect.HasValue 
                        ? (signal.IsCorrect.Value ? " ✓" : " ✗") 
                        : "";
                    Debug.Log($"[BCI] {signal.Type} applied | " +
                              $"net={signal.NetworkLatency:F0}ms, total={signal.TotalLatency:F0}ms{correctStr}");
                }
            }
        }

        private void CacheSource()
        {
            if (sourceBehaviour is IIntentSource intentSource)
            {
                _source = intentSource;
            }
            else if (sourceBehaviour != null)
            {
                Debug.LogError($"Source {sourceBehaviour.name} does not implement IIntentSource", this);
                _source = null;
            }
        }
    }
}

using UnityEngine;
using IntentFlow.Inputs;
using IntentFlow;

namespace IntentFlow.Inputs.MI
{
    public class MiSource : MonoBehaviour, IIntentSource
    {
        [Tooltip("IntentFlow websocket client (optional placeholder)")]
        public IntentFlowClient client;

        private IntentSignal? _pending;

        [ContextMenu("Debug Left")]
        public void DebugLeft() => _pending = new IntentSignal
        {
            Type = IntentType.Left,
            Confidence = 0.9f,
            Timestamp = Time.time,
        };

        [ContextMenu("Debug Right")]
        public void DebugRight() => _pending = new IntentSignal
        {
            Type = IntentType.Right,
            Confidence = 0.9f,
            Timestamp = Time.time,
        };

        public bool TryRead(out IntentSignal signal)
        {
            if (_pending.HasValue)
            {
                signal = _pending.Value;
                _pending = null;
                return true;
            }

            signal = default;
            return false;
        }
    }
}

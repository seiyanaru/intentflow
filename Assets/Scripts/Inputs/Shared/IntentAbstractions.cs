using UnityEngine;

namespace IntentFlow.Inputs
{
    public enum IntentType
    {
        Idle,
        Left,
        Right,
    }

    public struct IntentSignal
    {
        public IntentType Type;
        public float Confidence;
        public float Timestamp;
    }

    public interface IIntentSource
    {
        bool TryRead(out IntentSignal signal);
    }
}

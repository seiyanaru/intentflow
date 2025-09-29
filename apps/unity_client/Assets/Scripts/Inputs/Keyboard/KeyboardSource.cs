using UnityEngine;
using IntentFlow.Inputs;

namespace IntentFlow.Inputs.Keyboard
{
    public class KeyboardSource : MonoBehaviour, IIntentSource
    {
        [Header("Keyboard Mapping")]
        [Range(0f, 1f)] public float confidence = 1f;
        public KeyCode leftKey = KeyCode.A;
        public KeyCode rightKey = KeyCode.D;

        public bool TryRead(out IntentSignal signal)
        {
            signal = default;
            if (Input.GetKeyDown(leftKey))
            {
                signal = new IntentSignal
                {
                    Type = IntentType.Left,
                    Confidence = confidence,
                    Timestamp = Time.time,
                };
                return true;
            }

            if (Input.GetKeyDown(rightKey))
            {
                signal = new IntentSignal
                {
                    Type = IntentType.Right,
                    Confidence = confidence,
                    Timestamp = Time.time,
                };
                return true;
            }

            return false;
        }
    }
}

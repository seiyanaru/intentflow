using UnityEngine;
using IntentFlow.Inputs;

namespace IntentFlow.Inputs.Gates
{
    public class DecisionGate : MonoBehaviour, IIntentSource
    {
        [Header("Decision")]
        [SerializeField, Range(0f, 1f)] private float threshold = 0.6f;
        [SerializeField, Tooltip("Minimum seconds between accepted intents. 0 disables refractory.")]
        private float refractorySeconds = 0.25f;
        [SerializeField] private MonoBehaviour sourceBehaviour;

        private IIntentSource _source;
        private float _lastFireAt = -999f;

        private void Awake() => CacheSource();

        private void OnValidate()
        {
            if (!Application.isPlaying)
            {
                CacheSource();
            }
        }

        public bool TryRead(out IntentSignal signal)
        {
            signal = default;
            if (_source == null)
            {
                return false;
            }

            if (!_source.TryRead(out var raw))
            {
                return false;
            }

            if (raw.Confidence < threshold)
            {
                return false;
            }

            if (refractorySeconds > 0f && (Time.time - _lastFireAt) < refractorySeconds)
            {
                return false;
            }

            _lastFireAt = Time.time;
            signal = raw;
            return true;
        }

        public void SetSource(MonoBehaviour behaviour)
        {
            sourceBehaviour = behaviour;
            CacheSource();
            _lastFireAt = -999f;
        }

        public void SetSource(IIntentSource source)
        {
            _source = source;
            sourceBehaviour = source as MonoBehaviour;
            _lastFireAt = -999f;
        }

        private void CacheSource()
        {
            if (sourceBehaviour is IIntentSource intentSource)
            {
                _source = intentSource;
            }
            else if (sourceBehaviour != null)
            {
                Debug.LogError($"DecisionGate source {sourceBehaviour.name} does not implement IIntentSource", this);
                _source = null;
            }
        }
    }
}

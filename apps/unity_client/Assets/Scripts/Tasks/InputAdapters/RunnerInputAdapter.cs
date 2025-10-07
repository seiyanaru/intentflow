using System;
using UnityEngine;
using IntentFlow.Inputs;
using Tasks.Runner3Lane.Core;

namespace Tasks.Runner3Lane.InputAdapters
{
    public class RunnerInputAdapter : MonoBehaviour
    {
        [SerializeField] private RunnerController runner;
        [SerializeField] private MonoBehaviour sourceBehaviour;

        public int SuccessCount { get; private set; }
        public int FalseMoveCount { get; private set; }

        public event Action<IntentType> ActionTaken;

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
            if (!_enabled || _source == null || runner == null)
            {
                return;
            }

            if (_source.TryRead(out var signal))
            {
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

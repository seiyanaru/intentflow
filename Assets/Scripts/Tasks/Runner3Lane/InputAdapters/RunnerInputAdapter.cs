using UnityEngine;
using IntentFlow.Inputs;
using IntentFlow.Tasks.Runner3Lane.Core;

namespace IntentFlow.Tasks.Runner3Lane.InputAdapters
{
    public class RunnerInputAdapter : MonoBehaviour
    {
        [SerializeField] private RunnerController runner;
        [SerializeField] private MonoBehaviour sourceBehaviour;

        public int SuccessCount { get; private set; }
        public int FalseMoveCount { get; private set; }

        private IIntentSource _source;

        private void Awake()
        {
            if (sourceBehaviour is IIntentSource intentSource)
            {
                _source = intentSource;
            }
            else if (sourceBehaviour != null)
            {
                Debug.LogError($"Source {sourceBehaviour.name} does not implement IIntentSource", this);
            }

            if (!runner)
            {
                runner = FindObjectOfType<RunnerController>();
            }
        }

        private void Update()
        {
            if (_source == null || runner == null)
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
                        break;
                    case IntentType.Right:
                        runner.MoveRight();
                        SuccessCount++;
                        break;
                    default:
                        FalseMoveCount++;
                        break;
                }
            }
        }
    }
}

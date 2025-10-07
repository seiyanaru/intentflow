using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using IntentFlow.Inputs;
using Tasks.Runner3Lane.InputAdapters;

namespace Tasks.Runner3Lane.Core
{
    public class RunLogger : MonoBehaviour
    {
        [Serializable]
        public struct Sample
        {
            public float t;
            public int lane;
            public IntentType intent;
            public bool hit;
        }

        [SerializeField] private RunnerGameManager gameManager;
        [SerializeField] private RunnerInputAdapter inputAdapter;
        [SerializeField] private RunnerController runner;

        private readonly List<Sample> _samples = new List<Sample>();
        private bool _runActive;
        private float _runStartTime;

        private void Awake()
        {
            if (!gameManager)
            {
                gameManager = FindObjectOfType<RunnerGameManager>();
            }
            if (!inputAdapter)
            {
                inputAdapter = FindObjectOfType<RunnerInputAdapter>();
            }
            if (!runner)
            {
                runner = FindObjectOfType<RunnerController>();
            }
        }

        private void OnEnable()
        {
            if (gameManager != null)
            {
                gameManager.StateChanged += OnStateChanged;
            }
            if (inputAdapter != null)
            {
                inputAdapter.ActionTaken += OnActionTaken;
            }
            if (runner != null)
            {
                runner.Collided += OnRunnerCollided;
            }
        }

        private void OnDisable()
        {
            if (gameManager != null)
            {
                gameManager.StateChanged -= OnStateChanged;
            }
            if (inputAdapter != null)
            {
                inputAdapter.ActionTaken -= OnActionTaken;
            }
            if (runner != null)
            {
                runner.Collided -= OnRunnerCollided;
            }
        }

        private void OnActionTaken(IntentType intent)
        {
            if (!_runActive)
            {
                return;
            }

            _samples.Add(new Sample
            {
                t = Time.time - _runStartTime,
                lane = runner != null ? runner.CurrentLane : 0,
                intent = intent,
                hit = false,
            });
        }

        private void OnRunnerCollided(Obstacle obstacle)
        {
            if (!_runActive)
            {
                return;
            }

            _samples.Add(new Sample
            {
                t = Time.time - _runStartTime,
                lane = runner != null ? runner.CurrentLane : 0,
                intent = IntentType.Idle,
                hit = true,
            });
        }

        private void OnStateChanged(GameState state)
        {
            if (state == GameState.Running)
            {
                _samples.Clear();
                _runStartTime = Time.time;
                _runActive = true;
            }
            else if (_runActive && (state == GameState.Failed || state == GameState.Succeeded))
            {
                _runActive = false;
                WriteCsv(state);
            }
        }

        private void WriteCsv(GameState state)
        {
            try
            {
                var sb = new StringBuilder();
                sb.AppendLine("time,lane,intent,hit");
                foreach (var sample in _samples)
                {
                    sb.AppendLine($"{sample.t:F3},{sample.lane},{sample.intent},{(sample.hit ? 1 : 0)}");
                }

                sb.AppendLine();
                if (gameManager != null)
                {
                    sb.AppendLine($"summary,state,{state},score,{gameManager.Score},distance,{gameManager.Distance:F2},lives,{gameManager.Lives}");
                }

                var fileName = $"runlog_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
                var path = Path.Combine(Application.persistentDataPath, fileName);
                File.WriteAllText(path, sb.ToString());
                Debug.Log($"Run log saved to {path}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"RunLogger failed to write CSV: {ex.Message}");
            }
        }
    }
}

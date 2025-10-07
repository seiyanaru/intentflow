using UnityEngine;
using UnityEngine.UI;
using Tasks.Runner3Lane.Core;

namespace Tasks.Runner3Lane.UI
{
    public class Hud : MonoBehaviour
    {
        [SerializeField] private Text scoreText;
        [SerializeField] private Text distanceText;
        [SerializeField] private Text stateText;
        [SerializeField] private Button startButton;
        [SerializeField] private Button restartButton;

        private RunnerGameManager _manager;

        public void Bind(RunnerGameManager manager)
        {
            if (_manager != null)
            {
                _manager.ScoreChanged -= HandleScoreChanged;
                _manager.DistanceChanged -= HandleDistanceChanged;
                _manager.StateChanged -= HandleStateChanged;
            }

            _manager = manager;

            if (_manager != null)
            {
                _manager.ScoreChanged += HandleScoreChanged;
                _manager.DistanceChanged += HandleDistanceChanged;
                _manager.StateChanged += HandleStateChanged;
            }

            if (startButton)
            {
                startButton.onClick.RemoveListener(OnStartClicked);
                startButton.onClick.AddListener(OnStartClicked);
            }

            if (restartButton)
            {
                restartButton.onClick.RemoveListener(OnRestartClicked);
                restartButton.onClick.AddListener(OnRestartClicked);
            }

            RefreshAll();
        }

        public void SetPaused(bool paused)
        {
            if (paused)
            {
                if (stateText)
                {
                    stateText.text = "Paused";
                }
            }
            else
            {
                RefreshAll();
            }
        }

        private void OnDestroy()
        {
            Bind(null);
            if (startButton)
            {
                startButton.onClick.RemoveListener(OnStartClicked);
            }
            if (restartButton)
            {
                restartButton.onClick.RemoveListener(OnRestartClicked);
            }
        }

        private void OnStartClicked()
        {
            if (_manager != null)
            {
                _manager.StartRun();
            }
        }

        private void OnRestartClicked()
        {
            if (_manager != null)
            {
                _manager.Restart();
            }
        }

        private void HandleScoreChanged(int score)
        {
            if (scoreText)
            {
                scoreText.text = $"Score: {score}";
            }
        }

        private void HandleDistanceChanged(float distance)
        {
            if (distanceText)
            {
                distanceText.text = $"Distance: {distance:0.0}";
            }
        }

        private void HandleStateChanged(GameState state)
        {
            if (stateText)
            {
                var lives = _manager != null ? _manager.Lives : 0;
                stateText.text = $"{state} (Lives {lives})";
            }

            if (startButton)
            {
                startButton.interactable = state == GameState.Idle;
            }

            if (restartButton)
            {
                restartButton.interactable = state == GameState.Failed || state == GameState.Succeeded;
            }
        }

        private void RefreshAll()
        {
            if (_manager == null)
            {
                if (scoreText) scoreText.text = "Score: 0";
                if (distanceText) distanceText.text = "Distance: 0";
                if (stateText) stateText.text = GameState.Idle.ToString();
                return;
            }

            HandleScoreChanged(_manager.Score);
            HandleDistanceChanged(_manager.Distance);
            HandleStateChanged(_manager.State);
        }
    }
}

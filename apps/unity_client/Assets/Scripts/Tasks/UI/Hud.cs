using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Tasks.Runner3Lane.Core;

namespace Tasks.Runner3Lane.UI
{
    public class Hud : MonoBehaviour
    {
        [Header("Text Displays")]
        [SerializeField] private TMP_Text scoreText;
        [SerializeField] private TMP_Text distanceText;
        [SerializeField] private TMP_Text speedText;
        [SerializeField] private TMP_Text stateText;
        [SerializeField] private TMP_Text livesText;
        
        [Header("Life Icons")]
        [SerializeField] private Transform livesContainer;
        [SerializeField] private GameObject lifeIconPrefab;
        
        [Header("Buttons")]
        [SerializeField] private Button startButton;
        [SerializeField] private Button restartButton;
        
        [Header("Panels")]
        [SerializeField] private GameObject gameOverPanel;
        [SerializeField] private TMP_Text gameOverScoreText;
        [SerializeField] private TMP_Text gameOverMessageText;
        
        [Header("Settings")]
        [SerializeField] private string scoreFormat = "SCORE: {0:N0}";
        [SerializeField] private string distanceFormat = "DIST: {0:F0}m";
        [SerializeField] private string speedFormat = "SPEED: {0:F1}";
        [SerializeField] private Color successColor = new Color(0.3f, 0.9f, 0.3f);
        [SerializeField] private Color failColor = new Color(0.9f, 0.3f, 0.3f);

        private RunnerGameManager _manager;
        private RunnerController _runner;
        private GameObject[] _lifeIcons;

        private void Awake()
        {
            // Hide game over panel initially
            if (gameOverPanel != null)
            {
                gameOverPanel.SetActive(false);
            }
        }

        public void Bind(RunnerGameManager manager)
        {
            if (_manager != null)
            {
                _manager.ScoreChanged -= HandleScoreChanged;
                _manager.DistanceChanged -= HandleDistanceChanged;
                _manager.StateChanged -= HandleStateChanged;
            }

            _manager = manager;
            _runner = manager?.GetComponentInChildren<RunnerController>() 
                   ?? FindObjectOfType<RunnerController>();

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

            InitializeLivesDisplay();
            RefreshAll();
        }

        private void Update()
        {
            // Update speed display every frame
            if (speedText != null && _runner != null)
            {
                speedText.text = string.Format(speedFormat, _runner.ForwardSpeed);
            }
        }

        public void SetPaused(bool paused)
        {
            if (paused)
            {
                if (stateText)
                {
                    stateText.text = "⏸ PAUSED";
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
                scoreText.text = string.Format(scoreFormat, score);
            }
        }

        private void HandleDistanceChanged(float distance)
        {
            if (distanceText)
            {
                distanceText.text = string.Format(distanceFormat, distance);
            }
        }

        private void HandleStateChanged(GameState state)
        {
            UpdateStateDisplay(state);
            UpdateLivesDisplay();
            UpdateButtonStates(state);
            UpdateGameOverPanel(state);
        }

        private void UpdateStateDisplay(GameState state)
        {
            if (stateText == null) return;

            stateText.text = state switch
            {
                GameState.Idle => "PRESS SPACE TO START",
                GameState.Running => "▶ RUNNING",
                GameState.Paused => "⏸ PAUSED",
                GameState.Failed => "💀 GAME OVER",
                GameState.Succeeded => "🏆 COMPLETE!",
                _ => state.ToString()
            };

            stateText.color = state switch
            {
                GameState.Failed => failColor,
                GameState.Succeeded => successColor,
                _ => Color.white
            };
            }

        private void UpdateButtonStates(GameState state)
        {
            if (startButton)
            {
                startButton.gameObject.SetActive(state == GameState.Idle);
            }

            if (restartButton)
            {
                restartButton.gameObject.SetActive(state == GameState.Failed || state == GameState.Succeeded);
            }
        }

        private void UpdateGameOverPanel(GameState state)
        {
            if (gameOverPanel == null) return;

            bool showPanel = state == GameState.Failed || state == GameState.Succeeded;
            gameOverPanel.SetActive(showPanel);

            if (showPanel && _manager != null)
            {
                if (gameOverScoreText != null)
                {
                    gameOverScoreText.text = $"Final Score: {_manager.Score:N0}";
                }

                if (gameOverMessageText != null)
                {
                    if (state == GameState.Succeeded)
                    {
                        gameOverMessageText.text = "CONGRATULATIONS!";
                        gameOverMessageText.color = successColor;
                    }
                    else
                    {
                        gameOverMessageText.text = "GAME OVER";
                        gameOverMessageText.color = failColor;
                    }
                }
            }
        }

        private void InitializeLivesDisplay()
        {
            if (livesContainer == null || lifeIconPrefab == null) return;

            // Clear existing icons
            if (_lifeIcons != null)
            {
                foreach (var icon in _lifeIcons)
                {
                    if (icon != null) Destroy(icon);
                }
            }

            // Create new icons
            int maxLives = _manager != null ? _manager.Lives : 3;
            _lifeIcons = new GameObject[maxLives];

            for (int i = 0; i < maxLives; i++)
            {
                _lifeIcons[i] = Instantiate(lifeIconPrefab, livesContainer);
                _lifeIcons[i].SetActive(true);
            }
        }

        private void UpdateLivesDisplay()
        {
            if (_manager == null) return;

            int currentLives = _manager.Lives;

            // Update text display
            if (livesText != null)
            {
                livesText.text = $"♥ x {currentLives}";
            }

            // Update icon display
            if (_lifeIcons != null)
            {
                for (int i = 0; i < _lifeIcons.Length; i++)
                {
                    if (_lifeIcons[i] != null)
                    {
                        _lifeIcons[i].SetActive(i < currentLives);
                    }
                }
            }
        }

        private void RefreshAll()
        {
            if (_manager == null)
            {
                if (scoreText) scoreText.text = string.Format(scoreFormat, 0);
                if (distanceText) distanceText.text = string.Format(distanceFormat, 0f);
                if (speedText) speedText.text = string.Format(speedFormat, 0f);
                if (stateText) stateText.text = "PRESS SPACE TO START";
                if (livesText) livesText.text = "♥ x 3";
                return;
            }

            HandleScoreChanged(_manager.Score);
            HandleDistanceChanged(_manager.Distance);
            HandleStateChanged(_manager.State);
            UpdateLivesDisplay();
        }
    }
}

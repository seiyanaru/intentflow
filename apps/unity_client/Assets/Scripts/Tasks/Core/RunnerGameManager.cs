using System;
using UnityEngine;
using UnityEngine.SceneManagement;
using IntentFlow.Inputs;
using IntentFlow.Inputs.Gates;
using Tasks.Runner3Lane.InputAdapters;
using Tasks.Runner3Lane.UI;

namespace Tasks.Runner3Lane.Core
{
    public enum GameState
    {
        Idle,
        Running,
        Paused,
        Failed,
        Succeeded,
    }

    public enum InputMode
    {
        Keyboard,
        MI,
    }

    public class RunnerGameManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private RunnerController runner;
        [SerializeField] private ObstacleSpawner spawner;
        [SerializeField] private Hud hud;
        [SerializeField] private DespawnZone despawnZone;
        [SerializeField] private DifficultyProfile difficulty;
        [SerializeField] private RunnerInputAdapter inputAdapter;
        [SerializeField] private DecisionGate keyboardInput;
        [SerializeField] private DecisionGate miInput;

        [Header("Run Settings")]
        [SerializeField] private InputMode inputMode = InputMode.Keyboard;
        [SerializeField, Min(0f)] private float targetDistance = 200f;
        [SerializeField, Min(0f)] private float timeLimit = 60f;
        [SerializeField, Min(1)] private int startingLives = 3;

        public event Action<int> ScoreChanged;
        public event Action<float> DistanceChanged;
        public event Action<GameState> StateChanged;

        public GameState State { get; private set; } = GameState.Idle;
        public int Lives => _lives;
        public int Score => _score;
        public float Distance => _distance;

        private int _lives;
        private float _runStartTime;
        private float _startZ;
        private int _score;
        private float _distance;
        private float _pausedAt;

        private void Awake()
        {
            if (!runner)
            {
                runner = FindObjectOfType<RunnerController>();
            }
            if (!spawner)
            {
                spawner = FindObjectOfType<ObstacleSpawner>();
            }
            if (!hud)
            {
                hud = FindObjectOfType<Hud>();
            }
            if (!inputAdapter)
            {
                inputAdapter = FindObjectOfType<RunnerInputAdapter>();
            }
            if (!despawnZone)
            {
                despawnZone = FindObjectOfType<DespawnZone>();
            }
            if (despawnZone && spawner)
            {
                despawnZone.SetPool(spawner.Pool);
            }
        }

        private void OnEnable()
        {
            if (runner != null)
            {
                runner.Collided += OnRunnerCollided;
            }
            if (inputAdapter != null)
            {
                inputAdapter.ActionTaken += OnActionTaken;
            }
        }

        private void OnDisable()
        {
            if (runner != null)
            {
                runner.Collided -= OnRunnerCollided;
            }
            if (inputAdapter != null)
            {
                inputAdapter.ActionTaken -= OnActionTaken;
            }
        }

        private void Start()
        {
            _lives = startingLives;
            hud?.Bind(this);
            ApplyInputMode();
            if (spawner != null)
            {
                spawner.Active = false;
            }
            if (inputAdapter != null)
            {
                inputAdapter.Enabled = false;
            }
            SetState(GameState.Idle, true);
            NotifyScore();
            NotifyDistance();
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.Space) && State == GameState.Idle)
            {
                StartRun();
            }

            if (Input.GetKeyDown(KeyCode.Escape) && (State == GameState.Running || State == GameState.Paused))
            {
                Pause(State == GameState.Running);
            }

            if (Input.GetKeyDown(KeyCode.R) && (State == GameState.Failed || State == GameState.Succeeded))
            {
                Restart();
            }

            if (State != GameState.Running)
            {
                return;
            }

            var elapsed = Time.time - _runStartTime;
            if (runner != null)
            {
                _distance = Mathf.Max(0f, runner.transform.position.z - _startZ);
                DistanceChanged?.Invoke(_distance);
            }

            UpdateDifficulty(elapsed);

            if ((targetDistance > 0f && _distance >= targetDistance) ||
                (timeLimit > 0f && elapsed >= timeLimit))
            {
                CompleteRun();
            }
        }

        public void StartRun()
        {
            if (State == GameState.Running)
            {
                return;
            }

            _lives = startingLives;
            _runStartTime = Time.time;
            _startZ = runner ? runner.transform.position.z : 0f;
            _score = 0;
            _distance = 0f;
            NotifyScore();
            NotifyDistance();
            inputAdapter?.ResetStats();
            spawner?.ResetSpawner();
            ApplyInputMode();
            SetState(GameState.Running);
        }

        public void Pause(bool value)
        {
            if (State != GameState.Running && State != GameState.Paused)
            {
                return;
            }

            if (value)
            {
                if (State == GameState.Running)
                {
                    _pausedAt = Time.time;
                    SetState(GameState.Paused);
                }
            }
            else
            {
                if (State == GameState.Paused)
                {
                    var pauseDuration = Time.time - _pausedAt;
                    _runStartTime += pauseDuration;
                    SetState(GameState.Running);
                }
            }
        }

        public void Restart()
        {
            Time.timeScale = 1f;
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }

        private void UpdateDifficulty(float elapsed)
        {
            if (difficulty == null)
            {
                return;
            }

            if (spawner != null && difficulty.spawnIntervalOverTime != null && difficulty.spawnIntervalOverTime.length > 0)
            {
                var interval = difficulty.spawnIntervalOverTime.Evaluate(elapsed);
                spawner.SpawnInterval = Mathf.Max(difficulty.minSpawnInterval, interval);
            }

            if (runner != null && difficulty.forwardSpeedOverTime != null && difficulty.forwardSpeedOverTime.length > 0)
            {
                var speed = difficulty.forwardSpeedOverTime.Evaluate(elapsed);
                runner.ForwardSpeed = Mathf.Min(difficulty.maxForwardSpeed, Mathf.Max(0f, speed));
            }
        }

        private void OnActionTaken(IntentType intent)
        {
            if (State != GameState.Running)
            {
                return;
            }

            if (intent == IntentType.Left || intent == IntentType.Right)
            {
                _score++;
                NotifyScore();
            }
        }

        private void OnRunnerCollided(Obstacle obstacle)
        {
            if (State != GameState.Running)
            {
                spawner?.Release(obstacle);
                return;
            }

            _lives--;
            spawner?.Release(obstacle);
            if (_lives <= 0)
            {
                FailRun();
            }
            else
            {
                StateChanged?.Invoke(State);
            }
        }

        private void CompleteRun()
        {
            SetState(GameState.Succeeded);
        }

        private void FailRun()
        {
            SetState(GameState.Failed);
        }

        private void SetState(GameState newState, bool force = false)
        {
            if (!force && State == newState)
            {
                return;
            }

            State = newState;
            switch (newState)
            {
                case GameState.Running:
                    Time.timeScale = 1f;
                    if (spawner != null)
                    {
                        spawner.Active = true;
                    }
                    if (inputAdapter != null)
                    {
                        inputAdapter.Enabled = true;
                    }
                    ApplyInputMode();
                    hud?.SetPaused(false);
                    break;
                case GameState.Paused:
                    Time.timeScale = 0f;
                    if (spawner != null)
                    {
                        spawner.Active = false;
                    }
                    if (inputAdapter != null)
                    {
                        inputAdapter.Enabled = false;
                    }
                    hud?.SetPaused(true);
                    break;
                default:
                    Time.timeScale = 1f;
                    if (spawner != null)
                    {
                        spawner.Active = false;
                        spawner.ResetSpawner();
                    }
                    if (inputAdapter != null)
                    {
                        inputAdapter.Enabled = false;
                    }
                    hud?.SetPaused(false);
                    break;
            }

            StateChanged?.Invoke(State);
        }

        private void ApplyInputMode()
        {
            if (inputAdapter == null)
            {
                return;
            }

            IIntentSource selected = null;
            switch (inputMode)
            {
                case InputMode.Keyboard:
                    selected = keyboardInput;
                    if (keyboardInput) keyboardInput.enabled = true;
                    if (miInput) miInput.enabled = false;
                    break;
                case InputMode.MI:
                    selected = miInput;
                    if (miInput) miInput.enabled = true;
                    if (keyboardInput) keyboardInput.enabled = false;
                    break;
            }

            if (selected == null)
            {
                Debug.LogWarning("No intent source configured for current input mode", this);
            }

            inputAdapter.SetSource(selected);
        }

        private void NotifyScore() => ScoreChanged?.Invoke(_score);

        private void NotifyDistance() => DistanceChanged?.Invoke(_distance);

        private void OnValidate()
        {
            if (!Application.isPlaying)
            {
                ApplyInputMode();
            }
        }
    }
}

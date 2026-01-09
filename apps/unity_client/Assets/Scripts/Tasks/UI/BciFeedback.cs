using UnityEngine;
using UnityEngine.UI;
using TMPro;
using IntentFlow.Inputs;
using IntentFlow.Inputs.MI;

namespace Tasks.Runner3Lane.UI
{
    /// <summary>
    /// UI component that displays BCI prediction feedback including:
    /// - Current prediction (Left/Right/Idle)
    /// - Confidence bar
    /// - Connection status
    /// </summary>
    public class BciFeedback : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private TMP_Text predictionText;
        [SerializeField] private TMP_Text confidenceText;
        [SerializeField] private Image confidenceBar;
        [SerializeField] private Image connectionIndicator;
        [SerializeField] private TMP_Text connectionText;
        
        [Header("Colors")]
        [SerializeField] private Color leftColor = new Color(0.2f, 0.6f, 1f);
        [SerializeField] private Color rightColor = new Color(1f, 0.4f, 0.4f);
        [SerializeField] private Color idleColor = new Color(0.5f, 0.5f, 0.5f);
        [SerializeField] private Color connectedColor = new Color(0.3f, 0.9f, 0.3f);
        [SerializeField] private Color disconnectedColor = new Color(0.9f, 0.3f, 0.3f);
        
        [Header("Settings")]
        [SerializeField] private float confidenceDecaySpeed = 2f;
        [SerializeField] private float predictionDisplayDuration = 1.5f;
        
        [Header("Source")]
        [SerializeField] private MiSource miSource;
        
        private float _lastPredictionTime;
        private float _displayedConfidence;
        private IntentType _displayedIntent = IntentType.Idle;
        
        private void Start()
        {
            // Find MiSource if not assigned
            if (miSource == null)
            {
                miSource = FindObjectOfType<MiSource>();
            }
            
            if (miSource != null)
            {
                miSource.IntentReceived += OnIntentReceived;
                miSource.ConnectionStateChanged += OnConnectionStateChanged;
                
                // Initialize connection state
                OnConnectionStateChanged(miSource.IsConnected);
            }
            
            // Initialize UI
            SetPredictionDisplay(IntentType.Idle, 0f);
        }
        
        private void OnDestroy()
        {
            if (miSource != null)
            {
                miSource.IntentReceived -= OnIntentReceived;
                miSource.ConnectionStateChanged -= OnConnectionStateChanged;
            }
        }
        
        private void Update()
        {
            // Decay confidence over time after prediction
            float timeSincePrediction = Time.time - _lastPredictionTime;
            if (timeSincePrediction > predictionDisplayDuration)
            {
                _displayedConfidence = Mathf.Max(0f, _displayedConfidence - confidenceDecaySpeed * Time.deltaTime);
                
                if (_displayedConfidence <= 0.01f)
                {
                    _displayedIntent = IntentType.Idle;
                }
                
                UpdateConfidenceBar();
            }
        }
        
        private void OnIntentReceived(IntentSignal signal)
        {
            _displayedIntent = signal.Type;
            _displayedConfidence = signal.Confidence;
            _lastPredictionTime = Time.time;
            
            SetPredictionDisplay(signal.Type, signal.Confidence);
        }
        
        private void OnConnectionStateChanged(bool connected)
        {
            if (connectionIndicator != null)
            {
                connectionIndicator.color = connected ? connectedColor : disconnectedColor;
            }
            
            if (connectionText != null)
            {
                connectionText.text = connected ? "BCI: Connected" : "BCI: Disconnected";
                connectionText.color = connected ? connectedColor : disconnectedColor;
            }
        }
        
        private void SetPredictionDisplay(IntentType intent, float confidence)
        {
            // Update prediction text
            if (predictionText != null)
            {
                string intentStr = intent switch
                {
                    IntentType.Left => "← LEFT",
                    IntentType.Right => "RIGHT →",
                    _ => "IDLE"
                };
                
                predictionText.text = intentStr;
                predictionText.color = GetIntentColor(intent);
            }
            
            // Update confidence text
            if (confidenceText != null)
            {
                confidenceText.text = $"{confidence * 100:F0}%";
            }
            
            UpdateConfidenceBar();
        }
        
        private void UpdateConfidenceBar()
        {
            if (confidenceBar != null)
            {
                confidenceBar.fillAmount = _displayedConfidence;
                confidenceBar.color = GetIntentColor(_displayedIntent);
            }
        }
        
        private Color GetIntentColor(IntentType intent)
        {
            return intent switch
            {
                IntentType.Left => leftColor,
                IntentType.Right => rightColor,
                _ => idleColor
            };
        }
        
        /// <summary>
        /// Manually set the MiSource reference.
        /// </summary>
        public void SetSource(MiSource source)
        {
            // Unsubscribe from old source
            if (miSource != null)
            {
                miSource.IntentReceived -= OnIntentReceived;
                miSource.ConnectionStateChanged -= OnConnectionStateChanged;
            }
            
            miSource = source;
            
            // Subscribe to new source
            if (miSource != null)
            {
                miSource.IntentReceived += OnIntentReceived;
                miSource.ConnectionStateChanged += OnConnectionStateChanged;
                OnConnectionStateChanged(miSource.IsConnected);
            }
        }
    }
}



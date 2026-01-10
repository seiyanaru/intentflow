using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using IntentFlow.Inputs;
using Tasks.Runner3Lane.InputAdapters;

namespace Tasks.Runner3Lane.UI
{
    /// <summary>
    /// Displays BCI prediction results and latency information in the game UI.
    /// Shows: prediction, confidence, ground truth, latency breakdown, and statistics.
    /// </summary>
    public class BciLatencyDisplay : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private RunnerInputAdapter inputAdapter;
        
        [Header("UI Elements")]
        [SerializeField] private TextMeshProUGUI predictionText;
        [SerializeField] private TextMeshProUGUI latencyText;
        [SerializeField] private TextMeshProUGUI statsText;
        
        [Header("Settings")]
        [SerializeField] private bool showInConsole = true;
        [SerializeField] private int historySize = 20;
        
        // Statistics
        private int _totalCount;
        private int _correctCount;
        private List<double> _networkLatencies = new();
        private List<double> _totalLatencies = new();
        
        // Last signal info
        private IntentSignal? _lastSignal;
        
        private void Start()
        {
            if (inputAdapter == null)
            {
                inputAdapter = FindObjectOfType<RunnerInputAdapter>();
            }
            
            if (inputAdapter != null)
            {
                inputAdapter.SignalApplied += OnSignalApplied;
                Debug.Log("[BciLatencyDisplay] Connected to RunnerInputAdapter");
            }
            else
            {
                Debug.LogWarning("[BciLatencyDisplay] RunnerInputAdapter not found!");
            }
            
            // Ensure Rich Text is enabled on all text elements
            EnableRichText();
            
            UpdateUI();
        }
        
        private void EnableRichText()
        {
            if (predictionText != null) predictionText.richText = true;
            if (latencyText != null) latencyText.richText = true;
            if (statsText != null) statsText.richText = true;
        }
        
        private void OnDestroy()
        {
            if (inputAdapter != null)
            {
                inputAdapter.SignalApplied -= OnSignalApplied;
            }
        }
        
        private void OnSignalApplied(IntentSignal signal)
        {
            _lastSignal = signal;
            _totalCount++;
            
            // Debug: Check what we're receiving
            Debug.Log($"[BciLatencyDisplay] Signal received: Type={signal.Type}, TrueLabel='{signal.TrueLabel}', IsCorrect={signal.IsCorrect}");
            
            // Track correctness - check if TrueLabel matches prediction
            bool isCorrect = CheckCorrectness(signal);
            if (isCorrect)
            {
                _correctCount++;
            }
            
            // Track latencies
            if (signal.NetworkLatency >= 0)
            {
                _networkLatencies.Add(signal.NetworkLatency);
                if (_networkLatencies.Count > historySize)
                {
                    _networkLatencies.RemoveAt(0);
                }
            }
            
            if (signal.TotalLatency >= 0)
            {
                _totalLatencies.Add(signal.TotalLatency);
                if (_totalLatencies.Count > historySize)
                {
                    _totalLatencies.RemoveAt(0);
                }
            }
            
            UpdateUI();
            
            if (showInConsole)
            {
                LogSignalDetails(signal, isCorrect);
            }
        }
        
        /// <summary>Check if prediction matches ground truth.</summary>
        private bool CheckCorrectness(IntentSignal signal)
        {
            if (string.IsNullOrEmpty(signal.TrueLabel))
                return false;
            
            string trueLabel = signal.TrueLabel.ToLowerInvariant().Trim();
            
            return signal.Type switch
            {
                IntentType.Left => trueLabel == "left",
                IntentType.Right => trueLabel == "right",
                _ => false
            };
        }
        
        private void UpdateUI()
        {
            UpdatePredictionText();
            UpdateLatencyText();
            UpdateStatsText();
        }
        
        private void UpdatePredictionText()
        {
            if (predictionText == null) return;
            
            if (!_lastSignal.HasValue)
            {
                predictionText.text = "[PREDICTION]\nWaiting...";
                predictionText.color = Color.gray;
                return;
            }
            
            var s = _lastSignal.Value;
            bool isCorrect = CheckCorrectness(s);
            
            string prediction = s.Type == IntentType.Left ? "LEFT" : "RIGHT";
            int confPercent = Mathf.RoundToInt(s.Confidence * 100);
            string resultMark = isCorrect ? "OK" : "NG";
            
            var lines = new List<string>();
            lines.Add("[PREDICTION]");
            lines.Add($"Pred: {prediction}");
            lines.Add($"Conf: {confPercent}%");
            
            if (!string.IsNullOrEmpty(s.TrueLabel))
            {
                lines.Add($"True: {s.TrueLabel.ToUpper()}");
                lines.Add($"Result: {resultMark}");
            }
            
            if (s.TrialIdx > 0)
            {
                lines.Add($"Trial: #{s.TrialIdx}");
            }
            
            predictionText.text = string.Join("\n", lines);
            predictionText.color = isCorrect ? new Color(0.4f, 1f, 0.4f) : new Color(1f, 0.4f, 0.4f);
        }
        
        private void UpdateLatencyText()
        {
            if (latencyText == null) return;
            
            if (!_lastSignal.HasValue)
            {
                latencyText.text = "[LATENCY]\nWaiting...";
                latencyText.color = Color.gray;
                return;
            }
            
            var s = _lastSignal.Value;
            var lines = new List<string>();
            
            lines.Add("[LATENCY]");
            
            // Inference: EEG input -> classification
            if (s.InferenceMs > 0)
            {
                lines.Add($"Inference: {s.InferenceMs:F0} ms");
            }
            
            // Network: Server send -> Unity receive
            if (s.NetworkLatency >= 0)
            {
                lines.Add($"Network: {s.NetworkLatency:F0} ms");
            }
            
            // Total: Prediction -> Runner moved
            if (s.TotalLatency >= 0)
            {
                lines.Add($"Total: {s.TotalLatency:F0} ms");
            }
            
            // Average
            if (_totalLatencies.Count > 1)
            {
                double avgTotal = Average(_totalLatencies);
                lines.Add($"Avg: {avgTotal:F0} ms");
            }
            
            latencyText.text = string.Join("\n", lines);
            
            // Color: green < 100ms, yellow < 200ms, red > 200ms
            if (s.TotalLatency >= 0)
            {
                if (s.TotalLatency <= 100) latencyText.color = new Color(0.4f, 1f, 0.4f);
                else if (s.TotalLatency <= 200) latencyText.color = new Color(1f, 0.9f, 0.3f);
                else latencyText.color = new Color(1f, 0.4f, 0.4f);
            }
        }
        
        private void UpdateStatsText()
        {
            if (statsText == null) return;
            
            if (_totalCount == 0)
            {
                statsText.text = "[STATS]\nNo data";
                statsText.color = Color.gray;
                return;
            }
            
            float accuracy = (float)_correctCount / _totalCount * 100f;
            
            var lines = new List<string>();
            lines.Add("[STATS]");
            lines.Add($"Trials: {_totalCount}");
            lines.Add($"Correct: {_correctCount}");
            lines.Add($"Acc: {accuracy:F0}%");
            
            if (_totalLatencies.Count > 0)
            {
                double avgLat = Average(_totalLatencies);
                lines.Add($"AvgLat: {avgLat:F0} ms");
            }
            
            statsText.text = string.Join("\n", lines);
            
            // Color: green >= 70%, yellow >= 50%, red < 50%
            if (accuracy >= 70) statsText.color = new Color(0.4f, 1f, 0.4f);
            else if (accuracy >= 50) statsText.color = new Color(1f, 0.9f, 0.3f);
            else statsText.color = new Color(1f, 0.4f, 0.4f);
        }
        
        private double Average(List<double> values)
        {
            if (values.Count == 0) return 0;
            double sum = 0;
            foreach (var v in values) sum += v;
            return sum / values.Count;
        }
        
        private void LogSignalDetails(IntentSignal s, bool isCorrect)
        {
            string correctStr = isCorrect ? "CORRECT" : "WRONG";
            if (string.IsNullOrEmpty(s.TrueLabel)) correctStr = "N/A";
            
            Debug.Log($"[BCI] Trial {s.TrialIdx}: " +
                      $"{s.Type} vs {s.TrueLabel ?? "?"} = {correctStr} | " +
                      $"conf={s.Confidence:F0}%, net={s.NetworkLatency:F0}ms, total={s.TotalLatency:F0}ms");
        }
        
        /// <summary>Reset all statistics.</summary>
        public void ResetStats()
        {
            _totalCount = 0;
            _correctCount = 0;
            _networkLatencies.Clear();
            _totalLatencies.Clear();
            _lastSignal = null;
            UpdateUI();
        }
        
        /// <summary>Get current statistics as a dictionary.</summary>
        public Dictionary<string, object> GetStats()
        {
            return new Dictionary<string, object>
            {
                ["total"] = _totalCount,
                ["correct"] = _correctCount,
                ["accuracy"] = _totalCount > 0 ? (float)_correctCount / _totalCount : 0,
                ["avg_network_latency"] = _networkLatencies.Count > 0 ? Average(_networkLatencies) : -1,
                ["avg_total_latency"] = _totalLatencies.Count > 0 ? Average(_totalLatencies) : -1,
            };
        }
    }
}


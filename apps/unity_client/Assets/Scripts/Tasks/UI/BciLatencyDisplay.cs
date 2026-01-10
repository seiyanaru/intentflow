using System;
using System.Collections.Generic;
using UnityEngine;
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
            }
            else
            {
                Debug.LogWarning("[BciLatencyDisplay] RunnerInputAdapter not found!");
            }
            
            UpdateUI();
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
            
            // Track correctness
            if (signal.IsCorrect.HasValue && signal.IsCorrect.Value)
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
                LogSignalDetails(signal);
            }
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
                predictionText.text = "Waiting for BCI signal...";
                return;
            }
            
            var s = _lastSignal.Value;
            string predColor = s.Type == IntentType.Left ? "#4CAF50" : "#2196F3";  // Green for left, Blue for right
            string prediction = s.Type.ToString().ToUpper();
            string confStr = $"{s.Confidence * 100:F0}%";
            
            string result = $"<color={predColor}><b>{prediction}</b></color> ({confStr})";
            
            // Add ground truth comparison
            if (!string.IsNullOrEmpty(s.TrueLabel))
            {
                string trueLabel = s.TrueLabel.ToUpper();
                bool? isCorrect = s.IsCorrect;
                
                if (isCorrect.HasValue)
                {
                    string checkMark = isCorrect.Value ? "<color=#4CAF50>✓</color>" : "<color=#F44336>✗</color>";
                    result += $"\nTrue: {trueLabel} {checkMark}";
                }
                else
                {
                    result += $"\nTrue: {trueLabel}";
                }
            }
            
            if (s.TrialIdx > 0)
            {
                result += $"\n<size=80%>Trial #{s.TrialIdx}</size>";
            }
            
            predictionText.text = result;
        }
        
        private void UpdateLatencyText()
        {
            if (latencyText == null) return;
            
            if (!_lastSignal.HasValue)
            {
                latencyText.text = "Latency: --";
                return;
            }
            
            var s = _lastSignal.Value;
            var lines = new List<string>();
            
            // Network latency
            if (s.NetworkLatency >= 0)
            {
                string netColor = GetLatencyColor(s.NetworkLatency, 50, 100);
                lines.Add($"Network: <color={netColor}>{s.NetworkLatency:F0}</color> ms");
            }
            
            // Total latency
            if (s.TotalLatency >= 0)
            {
                string totalColor = GetLatencyColor(s.TotalLatency, 100, 200);
                lines.Add($"Total: <color={totalColor}>{s.TotalLatency:F0}</color> ms");
            }
            
            // Averages
            if (_networkLatencies.Count > 0)
            {
                double avgNet = Average(_networkLatencies);
                lines.Add($"<size=80%>Avg Net: {avgNet:F0} ms</size>");
            }
            
            if (_totalLatencies.Count > 0)
            {
                double avgTotal = Average(_totalLatencies);
                lines.Add($"<size=80%>Avg Total: {avgTotal:F0} ms</size>");
            }
            
            latencyText.text = string.Join("\n", lines);
        }
        
        private void UpdateStatsText()
        {
            if (statsText == null) return;
            
            if (_totalCount == 0)
            {
                statsText.text = "Stats: No data";
                return;
            }
            
            float accuracy = _totalCount > 0 ? (float)_correctCount / _totalCount * 100 : 0;
            string accColor = accuracy >= 70 ? "#4CAF50" : (accuracy >= 50 ? "#FFC107" : "#F44336");
            
            statsText.text = $"Predictions: {_totalCount}\n" +
                            $"Accuracy: <color={accColor}>{accuracy:F1}%</color>\n" +
                            $"({_correctCount}/{_totalCount})";
        }
        
        private string GetLatencyColor(double latency, double goodThreshold, double badThreshold)
        {
            if (latency <= goodThreshold) return "#4CAF50";  // Green
            if (latency <= badThreshold) return "#FFC107";   // Yellow
            return "#F44336";  // Red
        }
        
        private double Average(List<double> values)
        {
            if (values.Count == 0) return 0;
            double sum = 0;
            foreach (var v in values) sum += v;
            return sum / values.Count;
        }
        
        private void LogSignalDetails(IntentSignal s)
        {
            string correctStr = s.IsCorrect.HasValue 
                ? (s.IsCorrect.Value ? "CORRECT" : "WRONG") 
                : "N/A";
            
            Debug.Log($"[BciLatency] Trial {s.TrialIdx}: " +
                      $"pred={s.Type}, true={s.TrueLabel ?? "N/A"}, result={correctStr}, " +
                      $"conf={s.Confidence:F2}, " +
                      $"network={s.NetworkLatency:F0}ms, total={s.TotalLatency:F0}ms");
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


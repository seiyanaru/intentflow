using UnityEngine;

namespace IntentFlow.Inputs
{
    public enum IntentType
    {
        Idle,
        Left,
        Right,
    }

    /// <summary>
    /// Represents a BCI intent signal with detailed timing information for latency analysis.
    /// All timestamps are in milliseconds (Unix epoch).
    /// </summary>
    public struct IntentSignal
    {
        /// <summary>The predicted intent type.</summary>
        public IntentType Type;
        
        /// <summary>Model confidence [0, 1].</summary>
        public float Confidence;
        
        /// <summary>Legacy timestamp field (Unity time).</summary>
        public float Timestamp;
        
        // --- Detailed Timestamps for Latency Analysis ---
        
        /// <summary>When model prediction completed (server time, ms).</summary>
        public double PredictionTs;
        
        /// <summary>When server sent the message (server time, ms).</summary>
        public double SendTs;
        
        /// <summary>When Unity received the message (client time, ms).</summary>
        public double ReceiveTs;
        
        /// <summary>When Unity applied the action (client time, ms).</summary>
        public double ApplyTs;
        
        // --- Ground Truth Info ---
        
        /// <summary>Ground truth label if available ("left", "right", null).</summary>
        public string TrueLabel;
        
        /// <summary>Trial index from server.</summary>
        public int TrialIdx;
        
        // --- Computed Latencies ---
        
        /// <summary>Network latency: SendTs to ReceiveTs (ms).</summary>
        public double NetworkLatency => ReceiveTs > 0 && SendTs > 0 ? ReceiveTs - SendTs : -1;
        
        /// <summary>Total latency: PredictionTs to ApplyTs (ms).</summary>
        public double TotalLatency => ApplyTs > 0 && PredictionTs > 0 ? ApplyTs - PredictionTs : -1;
        
        /// <summary>Whether prediction matches ground truth.</summary>
        public bool? IsCorrect
        {
            get
            {
                if (string.IsNullOrEmpty(TrueLabel)) return null;
                var expected = TrueLabel.ToLowerInvariant() switch
                {
                    "left" => IntentType.Left,
                    "right" => IntentType.Right,
                    _ => IntentType.Idle
                };
                return Type == expected;
            }
        }
    }

    public interface IIntentSource
    {
        bool TryRead(out IntentSignal signal);
    }
}

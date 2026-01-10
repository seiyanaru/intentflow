using System;
using System.Collections;
using System.IO;
using UnityEngine;
using IntentFlow.Inputs;

namespace IntentFlow.Inputs.MI
{
    // #region Debug Helper
    public static class DebugLog
    {
        private static readonly string LogPath = "/workspace-cloud/seiya.narukawa/intentflow/.cursor/debug.log";
        public static void Log(string hyp, string loc, string msg, object data = null)
        {
            try
            {
                var json = $"{{\"hypothesisId\":\"{hyp}\",\"location\":\"{loc}\",\"message\":\"{msg}\",\"data\":\"{data}\",\"timestamp\":{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}}}";
                File.AppendAllText(LogPath, json + "\n");
            }
            catch { }
        }
    }
    // #endregion
    /// <summary>
    /// Motor Imagery input source that connects to the Python TTT Broadcaster
    /// via WebSocket and receives prediction intents.
    /// </summary>
    public class MiSource : MonoBehaviour, IIntentSource
    {
        [Header("WebSocket Settings")]
        [SerializeField] private string serverUrl = "ws://localhost:8765";
        [SerializeField] private bool autoConnect = true;
        [SerializeField] private float reconnectInterval = 3.0f;
        
        [Header("Debug")]
        [SerializeField] private bool logMessages = true;
        
        /// <summary>Event fired when an intent is received from the server.</summary>
        public event Action<IntentSignal> IntentReceived;
        
        /// <summary>Event fired when connection state changes.</summary>
        public event Action<bool> ConnectionStateChanged;
        
        /// <summary>Current connection state.</summary>
        public bool IsConnected { get; private set; }
        
        /// <summary>Last received confidence value.</summary>
        public float LastConfidence { get; private set; }
        
        /// <summary>Last received intent type.</summary>
        public IntentType LastIntentType { get; private set; }
        
        private IntentSignal? _pending;
        private WebSocketClient _wsClient;
        private bool _shouldReconnect = true;
        
        private void Awake()
        {
            _wsClient = new WebSocketClient();
            _wsClient.OnMessage += HandleMessage;
            _wsClient.OnOpen += HandleOpen;
            _wsClient.OnClose += HandleClose;
            _wsClient.OnError += HandleError;
        }
        
        private void Start()
        {
            if (autoConnect)
            {
                Connect();
            }
        }
        
        private void OnDestroy()
        {
            _shouldReconnect = false;
            Disconnect();
            
            if (_wsClient != null)
            {
                _wsClient.OnMessage -= HandleMessage;
                _wsClient.OnOpen -= HandleOpen;
                _wsClient.OnClose -= HandleClose;
                _wsClient.OnError -= HandleError;
            }
        }
        
        private void Update()
        {
            // Process WebSocket messages on main thread
            _wsClient?.DispatchMessageQueue();
        }
        
        /// <summary>Connect to the WebSocket server.</summary>
        public void Connect()
        {
            if (IsConnected) return;
            
            _shouldReconnect = true;
            StartCoroutine(ConnectCoroutine());
        }
        
        /// <summary>Disconnect from the WebSocket server.</summary>
        public void Disconnect()
        {
            _shouldReconnect = false;
            _wsClient?.Close();
        }
        
        private IEnumerator ConnectCoroutine()
        {
            if (logMessages)
            {
                Debug.Log($"[MiSource] Connecting to {serverUrl}...");
            }
            
            yield return _wsClient.Connect(serverUrl);
        }
        
        private void HandleOpen()
        {
            // #region agent log
            DebugLog.Log("C", "MiSource:HandleOpen", "WebSocket_connected", serverUrl);
            // #endregion
            IsConnected = true;
            ConnectionStateChanged?.Invoke(true);
            
            if (logMessages)
            {
                Debug.Log("[MiSource] Connected to TTT Broadcaster");
            }
        }
        
        private void HandleClose()
        {
            // #region agent log
            DebugLog.Log("C", "MiSource:HandleClose", "WebSocket_disconnected", null);
            // #endregion
            IsConnected = false;
            ConnectionStateChanged?.Invoke(false);
            
            if (logMessages)
            {
                Debug.Log("[MiSource] Disconnected from server");
            }
            
            // Auto-reconnect
            if (_shouldReconnect)
            {
                StartCoroutine(ReconnectCoroutine());
            }
        }
        
        private void HandleError(string error)
        {
            if (logMessages)
            {
                Debug.LogError($"[MiSource] WebSocket error: {error}");
            }
            
            // Auto-reconnect on connection failure
            if (_shouldReconnect && !IsConnected)
            {
                StartCoroutine(ReconnectCoroutine());
            }
        }
        
        private IEnumerator ReconnectCoroutine()
        {
            yield return new WaitForSeconds(reconnectInterval);
            
            if (_shouldReconnect && !IsConnected)
            {
                if (logMessages)
                {
                    Debug.Log("[MiSource] Attempting to reconnect...");
                }
                Connect();
            }
        }
        
        private void HandleMessage(string message)
        {
            // Record receive timestamp immediately
            double receiveTs = GetCurrentTimeMs();
            
            // #region agent log
            DebugLog.Log("C", "MiSource:HandleMessage", "raw_message_received", message?.Substring(0, Math.Min(100, message?.Length ?? 0)));
            // #endregion
            if (logMessages)
            {
                Debug.Log($"[MiSource] Raw message: {message}");
            }
            
            try
            {
                var intent = JsonUtility.FromJson<IntentMessage>(message);
                
                if (intent == null || intent.type != "intent")
                {
                    if (logMessages)
                    {
                        Debug.Log($"[MiSource] Ignored non-intent message: type={intent?.type}");
                    }
                    return;
                }
                
                // Map intent string to IntentType enum
                IntentType intentType = intent.intent.ToLowerInvariant() switch
                {
                    "left" => IntentType.Left,
                    "right" => IntentType.Right,
                    _ => IntentType.Idle
                };
                
                // Skip idle intents (we only care about left/right for lane control)
                if (intentType == IntentType.Idle)
                {
                    return;
                }
                
                // Create signal with detailed timing info
                var signal = new IntentSignal
                {
                    Type = intentType,
                    Confidence = intent.conf,
                    Timestamp = Time.time,
                    // Detailed timestamps for latency analysis
                    PredictionTs = intent.prediction_ts,
                    SendTs = intent.send_ts,
                    ReceiveTs = receiveTs,
                    ApplyTs = 0,  // Will be set by RunnerInputAdapter
                    // Ground truth info
                    TrueLabel = intent.true_label,
                    TrialIdx = intent.trial_idx,
                };
                
                _pending = signal;
                LastConfidence = intent.conf;
                LastIntentType = intentType;
                
                IntentReceived?.Invoke(signal);
                
                // Log with detailed debug info
                if (logMessages)
                {
                    // Debug: Show all parsed values
                    Debug.Log($"[MiSource] PARSED: pred_ts={intent.prediction_ts}, send_ts={intent.send_ts}, " +
                              $"recv_ts={receiveTs}, true_label='{intent.true_label}', trial={intent.trial_idx}");
                    
                    double networkLatency = signal.NetworkLatency;
                    string latencyStr = networkLatency >= 0 ? $", net={networkLatency:F0}ms" : $" (net calc failed: send={intent.send_ts}, recv={receiveTs})";
                    string trueStr = !string.IsNullOrEmpty(intent.true_label) ? $", true={intent.true_label}" : " (no true_label)";
                    Debug.Log($"[MiSource] Received: {intentType} (conf={intent.conf:F2}{trueStr}{latencyStr}) trial={intent.trial_idx}");
                }
            }
            catch (Exception e)
            {
                if (logMessages)
                {
                    Debug.LogWarning($"[MiSource] Failed to parse message: {e.Message}");
                }
            }
        }
        
        public bool TryRead(out IntentSignal signal)
        {
            if (_pending.HasValue)
            {
                signal = _pending.Value;
                _pending = null;
                return true;
            }
            
            signal = default;
            return false;
        }
        
        // Debug methods for testing
        [ContextMenu("Debug Left")]
        public void DebugLeft() => _pending = new IntentSignal
        {
            Type = IntentType.Left,
            Confidence = 0.9f,
            Timestamp = Time.time,
        };

        [ContextMenu("Debug Right")]
        public void DebugRight() => _pending = new IntentSignal
        {
            Type = IntentType.Right,
            Confidence = 0.9f,
            Timestamp = Time.time,
        };
        
        /// <summary>Intent message from Python server (protocol v2).</summary>
        [Serializable]
        private class IntentMessage
        {
            public string type;
            public string intent;
            public float conf;
            public double prediction_ts;  // When model prediction completed (ms)
            public double send_ts;        // When server sent the message (ms)
            public string true_label;     // Ground truth label
            public int trial_idx;         // Trial index
            public int protocol_version;
        }
        
        /// <summary>Get current time in milliseconds (Unix epoch).</summary>
        private static double GetCurrentTimeMs()
        {
            return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        }
    }
    
    /// <summary>
    /// Simple WebSocket client wrapper for Unity.
    /// Uses Unity's built-in WebSocket support where available,
    /// or provides a fallback implementation.
    /// </summary>
    public class WebSocketClient
    {
        public event Action OnOpen;
        public event Action OnClose;
        public event Action<string> OnMessage;
        public event Action<string> OnError;
        
        private System.Collections.Generic.Queue<string> _messageQueue = new();
        private object _lock = new object();
        
#if UNITY_WEBGL && !UNITY_EDITOR
        // WebGL implementation using JavaScript interop
        [System.Runtime.InteropServices.DllImport("__Internal")]
        private static extern int WebSocket_Create(string url);
        [System.Runtime.InteropServices.DllImport("__Internal")]
        private static extern void WebSocket_Close(int id);
        [System.Runtime.InteropServices.DllImport("__Internal")]
        private static extern void WebSocket_Send(int id, string message);
        
        private int _socketId = -1;
        
        public IEnumerator Connect(string url)
        {
            _socketId = WebSocket_Create(url);
            yield return new WaitForSeconds(0.5f);
            OnOpen?.Invoke();
        }
        
        public void Close()
        {
            if (_socketId >= 0)
            {
                WebSocket_Close(_socketId);
                _socketId = -1;
            }
            OnClose?.Invoke();
        }
        
        public void Send(string message)
        {
            if (_socketId >= 0)
            {
                WebSocket_Send(_socketId, message);
            }
        }
#else
        // Standalone/Editor implementation using System.Net.WebSockets
        private System.Net.WebSockets.ClientWebSocket _socket;
        private System.Threading.CancellationTokenSource _cts;
        private bool _isConnected;
        
        public IEnumerator Connect(string url)
        {
            _socket = new System.Net.WebSockets.ClientWebSocket();
            _cts = new System.Threading.CancellationTokenSource();
            
            // Note: Connection and Upgrade headers are set automatically by ClientWebSocket
            // Do NOT set them manually as it causes conflicts
            
            System.Threading.Tasks.Task connectTask = null;
            try
            {
                connectTask = _socket.ConnectAsync(new Uri(url), _cts.Token);
            }
            catch (Exception e)
            {
                OnError?.Invoke($"Connection setup failed: {e.Message}");
                yield break;
            }
            
            float timeout = 10f;
            float elapsed = 0f;
            while (!connectTask.IsCompleted && elapsed < timeout)
            {
                elapsed += UnityEngine.Time.deltaTime;
                yield return null;
            }
            
            if (!connectTask.IsCompleted)
            {
                OnError?.Invoke("Connection timeout");
                yield break;
            }
            
            if (connectTask.IsFaulted)
            {
                var ex = connectTask.Exception?.InnerException ?? connectTask.Exception;
                var fullError = $"{ex?.GetType().Name}: {ex?.Message}";
                if (ex?.InnerException != null)
                {
                    fullError += $" -> {ex.InnerException.GetType().Name}: {ex.InnerException.Message}";
                }
                OnError?.Invoke(fullError);
                yield break;
            }
            
            _isConnected = true;
            OnOpen?.Invoke();
            
            // Start receive loop
            _ = ReceiveLoop();
        }
        
        private async System.Threading.Tasks.Task ReceiveLoop()
        {
            var buffer = new byte[4096];
            var messageBuffer = new System.Collections.Generic.List<byte>();
            
            try
            {
                while (_isConnected && _socket.State == System.Net.WebSockets.WebSocketState.Open)
                {
                    var result = await _socket.ReceiveAsync(
                        new ArraySegment<byte>(buffer), 
                        _cts.Token
                    );
                    
                    if (result.MessageType == System.Net.WebSockets.WebSocketMessageType.Close)
                    {
                        _isConnected = false;
                        OnClose?.Invoke();
                        break;
                    }
                    
                    messageBuffer.AddRange(new ArraySegment<byte>(buffer, 0, result.Count));
                    
                    if (result.EndOfMessage)
                    {
                        var message = System.Text.Encoding.UTF8.GetString(messageBuffer.ToArray());
                        messageBuffer.Clear();
                        
                        lock (_lock)
                        {
                            _messageQueue.Enqueue(message);
                        }
                    }
                }
            }
            catch (Exception e)
            {
                if (_isConnected)
                {
                    OnError?.Invoke(e.Message);
                    _isConnected = false;
                    OnClose?.Invoke();
                }
            }
        }
        
        public void Close()
        {
            _isConnected = false;
            _cts?.Cancel();
            
            try
            {
                if (_socket?.State == System.Net.WebSockets.WebSocketState.Open)
                {
                    _ = _socket.CloseAsync(
                        System.Net.WebSockets.WebSocketCloseStatus.NormalClosure,
                        "Client closing",
                        System.Threading.CancellationToken.None
                    );
                }
            }
            catch { }
            
            OnClose?.Invoke();
        }
        
        public void Send(string message)
        {
            if (_socket?.State != System.Net.WebSockets.WebSocketState.Open) return;
            
            var bytes = System.Text.Encoding.UTF8.GetBytes(message);
            _ = _socket.SendAsync(
                new ArraySegment<byte>(bytes),
                System.Net.WebSockets.WebSocketMessageType.Text,
                true,
                _cts?.Token ?? System.Threading.CancellationToken.None
            );
        }
#endif
        
        public void DispatchMessageQueue()
        {
            lock (_lock)
            {
                while (_messageQueue.Count > 0)
                {
                    var message = _messageQueue.Dequeue();
                    OnMessage?.Invoke(message);
                }
            }
        }
    }
}

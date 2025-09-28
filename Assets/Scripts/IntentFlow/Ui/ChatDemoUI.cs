using System;
using System.Text;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace IntentFlow.Ui
{
    public class ChatDemoUI : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private TMP_InputField input;
        [SerializeField] private TMP_Text logText;
        [SerializeField] private Button sendButton;
        [SerializeField] private ScrollRect scroll;

        [Header("Client")]
        [SerializeField] private IntentFlow.IntentFlowClient client;
        [SerializeField] private bool autoScroll = true;

        private readonly StringBuilder _logBuilder = new StringBuilder();
        private CancellationTokenSource _cts;
        private bool _isSending;
        private UnityAction<string> _inputChangedHandler;

        private void Awake()
        {
            _cts = new CancellationTokenSource();
            if (sendButton != null)
            {
                sendButton.onClick.AddListener(OnSendClicked);
            }
            if (input != null)
            {
                _inputChangedHandler = _ => UpdateButtonState();
                input.onValueChanged.AddListener(_inputChangedHandler);
            }
            UpdateButtonState();
        }

        private void OnDestroy()
        {
            if (sendButton != null)
            {
                sendButton.onClick.RemoveListener(OnSendClicked);
            }
            if (input != null && _inputChangedHandler != null)
            {
                input.onValueChanged.RemoveListener(_inputChangedHandler);
            }
            try
            {
                _cts?.Cancel();
            }
            catch (Exception) { }
            _cts?.Dispose();
            _cts = null;
        }

        private void UpdateButtonState()
        {
            if (sendButton == null || input == null)
            {
                return;
            }
            var hasText = !string.IsNullOrWhiteSpace(input.text);
            sendButton.interactable = !_isSending && hasText;
        }

        private async void OnSendClicked()
        {
            if (input == null || client == null)
            {
                return;
            }

            var text = input.text?.Trim();
            if (string.IsNullOrEmpty(text))
            {
                UpdateButtonState();
                return;
            }

            input.text = string.Empty;
            UpdateButtonState();
            LogLine($"You: {text}");

            try
            {
                _isSending = true;
                UpdateButtonState();
                var response = await client.SendEchoAsync(text, _cts.Token);
                LogLine($"Bot: {response.reply}");
            }
            catch (OperationCanceledException)
            {
                LogLine("Error: Request cancelled");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[ChatDemoUI] Send failed: {ex}");
                LogLine($"Error: {ex.Message}");
            }
            finally
            {
                _isSending = false;
                UpdateButtonState();
            }
        }

        private void LogLine(string line)
        {
            if (logText == null)
            {
                return;
            }

            if (_logBuilder.Length > 0)
            {
                _logBuilder.AppendLine();
            }
            _logBuilder.Append(line);
            logText.text = _logBuilder.ToString();

            if (autoScroll && scroll != null)
            {
                Canvas.ForceUpdateCanvases();
                scroll.verticalNormalizedPosition = 0f;
            }
        }
    }
}

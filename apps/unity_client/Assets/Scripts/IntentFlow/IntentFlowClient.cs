using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace IntentFlow
{
    public class IntentFlowClient : MonoBehaviour
    {
        [Header("Endpoint")]
        [SerializeField] private string baseUrl = "http://localhost:8000";
        [SerializeField] private string endpoint = "/echo";
        [SerializeField] private int timeoutSeconds = 10;

        private CancellationTokenSource _cts;

        private void Awake()
        {
            _cts = new CancellationTokenSource();
        }

        public async Task<EchoResponse> SendEchoAsync(string text, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(text))
            {
                throw new ArgumentException("Text must not be empty", nameof(text));
            }

            var request = new EchoRequest { text = text };
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token, cancellationToken);
            return await IntentApi.PostEchoAsync(baseUrl, endpoint, request, timeoutSeconds, linked.Token);
        }

        private void OnDestroy()
        {
            if (_cts != null)
            {
                try
                {
                    _cts.Cancel();
                }
                catch (Exception) { }
                _cts.Dispose();
                _cts = null;
            }
        }
    }
}

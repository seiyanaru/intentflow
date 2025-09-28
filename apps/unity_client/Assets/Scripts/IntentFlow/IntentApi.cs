using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace IntentFlow
{
    public static class IntentApi
    {
        public static async Task<EchoResponse> PostEchoAsync(
            string baseUrl,
            string endpoint,
            EchoRequest request,
            int timeoutSec,
            CancellationToken cancellationToken)
        {
            if (string.IsNullOrEmpty(baseUrl))
            {
                throw new ArgumentException("Base URL must be set", nameof(baseUrl));
            }

            if (request == null)
            {
                throw new ArgumentNullException(nameof(request));
            }

            var uri = new Uri(new Uri(baseUrl), endpoint ?? string.Empty);
            var json = JsonUtility.ToJson(request);
            var payload = Encoding.UTF8.GetBytes(json ?? "{}");

            using var uwr = new UnityWebRequest(uri, UnityWebRequest.kHttpVerbPOST)
            {
                downloadHandler = new DownloadHandlerBuffer(),
                uploadHandler = new UploadHandlerRaw(payload)
            };

            uwr.SetRequestHeader("Content-Type", "application/json; charset=utf-8");
            var timeoutSeconds = Mathf.Max(1, timeoutSec);
            uwr.timeout = timeoutSeconds;

            using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);
            using (linkedCts.Token.Register(() =>
            {
                if (!uwr.isDone)
                {
                    try
                    {
                        uwr.Abort();
                    }
                    catch (Exception) { }
                }
            }))
            {
                var operation = uwr.SendWebRequest();
                while (!operation.isDone)
                {
                    if (linkedCts.Token.IsCancellationRequested)
                    {
                        throw new OperationCanceledException(linkedCts.Token);
                    }

                    await Task.Yield();
                }
            }

            if (timeoutCts.IsCancellationRequested)
            {
                throw new TimeoutException($"Request timed out after {timeoutSeconds} seconds");
            }

            if (cancellationToken.IsCancellationRequested)
            {
                throw new OperationCanceledException(cancellationToken);
            }

            if (uwr.result != UnityWebRequest.Result.Success)
            {
                var message = uwr.downloadHandler != null ? uwr.downloadHandler.text : string.Empty;
                throw new Exception($"HTTP {(int)uwr.responseCode}: {uwr.error} {message}");
            }

            var body = uwr.downloadHandler.text;
            if (string.IsNullOrEmpty(body))
            {
                throw new Exception("Empty response body");
            }

            EchoResponse response;
            try
            {
                response = JsonUtility.FromJson<EchoResponse>(body);
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to parse response: {ex.Message}\nBody: {body}");
            }

            if (response == null)
            {
                throw new Exception("Response deserialized to null");
            }

            return response;
        }
    }
}

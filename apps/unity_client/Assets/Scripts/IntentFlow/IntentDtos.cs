using System;

namespace IntentFlow
{
    [Serializable]
    public class EchoRequest
    {
        public string text = string.Empty;
    }

    [Serializable]
    public class EchoResponse
    {
        public string reply = string.Empty;
    }
}

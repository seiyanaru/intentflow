# Minimal Unity HTTP Echo Client

This sample demonstrates how to send a JSON payload from Unity to an HTTP endpoint and display the reply in a simple chat UI.

## Project Layout
```
Assets/
  Scenes/
    Demo.unity            # wire the scene following the steps below
  Scripts/
    IntentFlow/
      IntentDtos.cs       # Request/response DTOs
      IntentApi.cs        # UnityWebRequest helper
      IntentFlowClient.cs # Slim wrapper exposed to the Inspector
      Ui/ChatDemoUI.cs    # UI binding logic
```

## Scene Wiring (Demo.unity)
1. Create a **Canvas (Screen Space - Overlay)**.
2. Under the canvas, add:
   - **InputRow** (empty GameObject):
     - `TMP_InputField` (placeholder "Type message…"). Assign to `ChatDemoUI.input`.
     - `Button` (text "Send"). Assign to `ChatDemoUI.sendButton`.
   - **ScrollView** (UI → Scroll View): replace Content text with a `TMP_Text` and assign it to `ChatDemoUI.logText`. Assign the `ScrollRect` reference to `ChatDemoUI.scroll`.
3. Create an empty GameObject named **IntentFlow** and add `IntentFlowClient`.
4. Create an empty GameObject named **UI** and add `ChatDemoUI`. Drag the references above and set `autoScroll = true`.
5. Save the scene as `Assets/Scenes/Demo.unity`.

_Default IntentFlowClient settings:_
```
Base Url   = http://localhost:8000
Endpoint   = /echo
Timeout    = 10 (seconds)
```

## Mock Echo Server
Choose one of the following quick servers:

### Node.js (Express)
```bash
npm init -y
npm install express
cat <<'JS' > server.js
const express = require('express');
const app = express();
app.use(express.json());
app.post('/echo', (req, res) => {
  const t = (req.body && req.body.text) || '';
  res.json({ reply: `Echo: ${t}` });
});
app.listen(8000, () => console.log('Echo server on http://localhost:8000'));
JS
node server.js
```

### Python (Flask)
```bash
pip install flask
cat <<'PY' > server.py
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.post('/echo')
def echo():
    text = (request.get_json() or {}).get('text', '')
    return jsonify(reply=f"Echo: {text}")

app.run('0.0.0.0', 8000)
PY
python server.py
```

## Play Mode Test
1. Start the mock server (`node server.js` or `python server.py`).
2. Open `Demo.unity` in Unity 2022.3 LTS.
3. Enter Play mode.
4. Type `hello` in the input field and press **Send**.
5. The log output should show:
   ```
   You: hello
   Bot: Echo: hello
   ```

If the server is unavailable, the UI prints `Error: ...` along with console logs, and the Send button re-enables after the request completes.

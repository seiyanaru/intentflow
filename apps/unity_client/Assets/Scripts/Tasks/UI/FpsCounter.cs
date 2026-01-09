using UnityEngine;
using TMPro;

namespace Tasks.Runner3Lane.UI
{
    /// <summary>
    /// Displays FPS (frames per second) counter.
    /// </summary>
    public class FpsCounter : MonoBehaviour
    {
        [Header("Display")]
        [SerializeField] private TMP_Text fpsText;
        [SerializeField] private string format = "FPS: {0:F0}";
        
        [Header("Settings")]
        [SerializeField] private float updateInterval = 0.5f;
        
        [Header("Color Thresholds")]
        [SerializeField] private float goodFps = 55f;
        [SerializeField] private float okFps = 30f;
        [SerializeField] private Color goodColor = new Color(0.3f, 0.9f, 0.3f);
        [SerializeField] private Color okColor = new Color(0.9f, 0.9f, 0.3f);
        [SerializeField] private Color badColor = new Color(0.9f, 0.3f, 0.3f);
        
        private float _deltaTime;
        private float _timer;
        
        private void Update()
        {
            _deltaTime += (Time.unscaledDeltaTime - _deltaTime) * 0.1f;
            _timer += Time.unscaledDeltaTime;
            
            if (_timer >= updateInterval)
            {
                _timer = 0f;
                UpdateDisplay();
            }
        }
        
        private void UpdateDisplay()
        {
            if (fpsText == null) return;
            
            float fps = 1.0f / _deltaTime;
            fpsText.text = string.Format(format, fps);
            
            // Color based on performance
            if (fps >= goodFps)
            {
                fpsText.color = goodColor;
            }
            else if (fps >= okFps)
            {
                fpsText.color = okColor;
            }
            else
            {
                fpsText.color = badColor;
            }
        }
    }
}


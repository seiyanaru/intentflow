using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    /// <summary>
    /// Adds visual effects to obstacles (rotation, pulse, glow).
    /// Attach this alongside the Obstacle component.
    /// </summary>
    public class ObstacleVisual : MonoBehaviour
    {
        [Header("Rotation")]
        [SerializeField] private bool enableRotation = true;
        [SerializeField] private Vector3 rotationSpeed = new Vector3(0f, 90f, 0f);
        
        [Header("Pulse")]
        [SerializeField] private bool enablePulse = false;
        [SerializeField] private float pulseSpeed = 2f;
        [SerializeField] private float pulseAmount = 0.1f;
        
        [Header("Hover")]
        [SerializeField] private bool enableHover = true;
        [SerializeField] private float hoverSpeed = 2f;
        [SerializeField] private float hoverHeight = 0.2f;
        
        private Vector3 _baseScale;
        private Vector3 _basePosition;
        private float _randomOffset;
        
        private void OnEnable()
        {
            _baseScale = transform.localScale;
            _basePosition = transform.localPosition;
            _randomOffset = Random.value * Mathf.PI * 2f;
        }
        
        private void Update()
        {
            // Rotation
            if (enableRotation)
            {
                transform.Rotate(rotationSpeed * Time.deltaTime);
            }
            
            // Pulse scale
            if (enablePulse)
            {
                float pulse = 1f + Mathf.Sin((Time.time + _randomOffset) * pulseSpeed) * pulseAmount;
                transform.localScale = _baseScale * pulse;
            }
            
            // Hover up/down
            if (enableHover)
            {
                float hover = Mathf.Sin((Time.time + _randomOffset) * hoverSpeed) * hoverHeight;
                var pos = _basePosition;
                pos.y += hover;
                transform.localPosition = pos;
            }
        }
        
        private void OnDisable()
        {
            // Reset to base state
            transform.localScale = _baseScale;
            transform.localPosition = _basePosition;
            transform.localRotation = Quaternion.identity;
        }
    }
}


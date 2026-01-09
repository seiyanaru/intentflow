using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    /// <summary>
    /// Handles visual effects for the runner (tilt, bounce, trail).
    /// Attach to a child object of the Runner that contains the visual mesh.
    /// </summary>
    public class RunnerVisual : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private RunnerController runner;
        
        [Header("Tilt Effect")]
        [SerializeField] private bool enableTilt = true;
        [SerializeField] private float maxTiltAngle = 15f;
        [SerializeField] private float tiltSpeed = 8f;
        
        [Header("Bounce Effect")]
        [SerializeField] private bool enableBounce = true;
        [SerializeField] private float bounceSpeed = 8f;
        [SerializeField] private float bounceAmount = 0.05f;
        
        [Header("Speed Stretch")]
        [SerializeField] private bool enableStretch = true;
        [SerializeField] private float stretchAmount = 0.1f;
        [SerializeField] private float baseSpeed = 10f;
        
        private float _targetTilt;
        private float _currentTilt;
        private int _lastLane = 1;
        private Vector3 _baseScale;
        
        private void Start()
        {
            if (runner == null)
            {
                runner = GetComponentInParent<RunnerController>();
            }
            _baseScale = transform.localScale;
            _lastLane = runner != null ? runner.CurrentLane : 1;
        }
        
        private void Update()
        {
            if (runner == null) return;
            
            UpdateTilt();
            UpdateBounce();
            UpdateStretch();
        }
        
        private void UpdateTilt()
        {
            if (!enableTilt) return;
            
            int currentLane = runner.CurrentLane;
            
            // Detect lane change direction
            if (currentLane != _lastLane)
            {
                _targetTilt = (currentLane > _lastLane) ? -maxTiltAngle : maxTiltAngle;
                _lastLane = currentLane;
            }
            else
            {
                // Return to neutral
                _targetTilt = 0f;
            }
            
            // Smooth tilt
            _currentTilt = Mathf.Lerp(_currentTilt, _targetTilt, tiltSpeed * Time.deltaTime);
            
            // Apply rotation (tilt on Z axis)
            var euler = transform.localEulerAngles;
            euler.z = _currentTilt;
            transform.localEulerAngles = euler;
        }
        
        private void UpdateBounce()
        {
            if (!enableBounce) return;
            
            // Simple sine wave bounce
            float bounce = Mathf.Sin(Time.time * bounceSpeed) * bounceAmount;
            var pos = transform.localPosition;
            pos.y = bounce;
            transform.localPosition = pos;
        }
        
        private void UpdateStretch()
        {
            if (!enableStretch) return;
            
            // Stretch based on speed
            float speedFactor = runner.ForwardSpeed / baseSpeed;
            float stretch = 1f + (speedFactor - 1f) * stretchAmount;
            
            var scale = _baseScale;
            scale.z *= stretch;
            scale.x /= Mathf.Sqrt(stretch); // Compensate width
            transform.localScale = scale;
        }
        
        /// <summary>
        /// Play hit reaction effect.
        /// </summary>
        public void PlayHitEffect()
        {
            // Could trigger animation, particles, etc.
            StartCoroutine(HitFlash());
        }
        
        private System.Collections.IEnumerator HitFlash()
        {
            var renderer = GetComponent<Renderer>();
            if (renderer == null) yield break;
            
            var originalColor = renderer.material.color;
            renderer.material.color = Color.red;
            
            for (int i = 0; i < 3; i++)
            {
                renderer.enabled = false;
                yield return new WaitForSeconds(0.1f);
                renderer.enabled = true;
                yield return new WaitForSeconds(0.1f);
            }
            
            renderer.material.color = originalColor;
        }
    }
}


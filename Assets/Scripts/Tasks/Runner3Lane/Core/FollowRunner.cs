using UnityEngine;
// RunnerController が定義されている名前空間を参照
using IntentFlow.Tasks.Runner3Lane.Core;

namespace Tasks.Runner3Lane.Core
{
    /// <summary>
    /// Smoothly follows a runner using a configurable offset and look-ahead.
    /// </summary>
    public class FollowRunner : MonoBehaviour
    {
        [SerializeField, Tooltip("Target transform to follow.")]
        private Transform target;

        [SerializeField, Tooltip("World-space offset from the target.")]
        private Vector3 offset = new Vector3(0f, 6f, -12f);

        [SerializeField, Tooltip("Lerp factor for following movement.")]
        private float followLerp = 5f;

        [SerializeField, Tooltip("Forward offset applied when looking at the target.")]
        private float lookAhead = 10f;

        [SerializeField, Tooltip("Freeze X axis movement when enabled.")]
        private bool freezeX;

        [SerializeField, Tooltip("Freeze Y axis movement when enabled.")]
        private bool freezeY;

        [SerializeField, Tooltip("Freeze Z axis movement when enabled.")]
        private bool freezeZ;

        private void Reset()
        {
            offset = new Vector3(0f, 6f, -12f);
            followLerp = 5f;
            lookAhead = 10f;
            freezeX = false;
            freezeY = false;
            freezeZ = false;
        }

        private void Awake()
        {
            if (target == null)
            {
                var runner = FindObjectOfType<RunnerController>();
                if (runner != null)
                {
                    target = runner.transform;
                }
            }
        }

        private void LateUpdate()
        {
            if (target == null) return;

            var current = transform.position;
            var desired = target.position + offset;

            var lerpT = followLerp * Time.deltaTime;
            if (lerpT > 1f) lerpT = 1f;

            var newPos = Vector3.Lerp(current, desired, lerpT);

            if (freezeX) newPos.x = current.x;
            if (freezeY) newPos.y = current.y;
            if (freezeZ) newPos.z = current.z;

            transform.position = newPos;

            var lookAt = target.position;
            lookAt.z += lookAhead;
            transform.LookAt(lookAt);
        }
    }
}

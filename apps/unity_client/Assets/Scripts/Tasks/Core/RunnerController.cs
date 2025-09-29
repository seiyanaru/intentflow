using UnityEngine;

namespace IntentFlow.Tasks.Runner3Lane.Core
{
    public class RunnerController : MonoBehaviour
    {
        [Header("Movement")]
        public float forwardSpeed = 6f;
        public float laneWidth = 2f;
        public float laneChangeLerp = 12f;

        private int _currentLane = 1;

        private float TargetX => (_currentLane - 1) * laneWidth;

        public void MoveLeft()
        {
            if (_currentLane > 0)
            {
                _currentLane--;
            }
        }

        public void MoveRight()
        {
            if (_currentLane < 2)
            {
                _currentLane++;
            }
        }

        private void Update()
        {
            transform.position += Vector3.forward * (forwardSpeed * Time.deltaTime);
            var pos = transform.position;
            pos.x = Mathf.Lerp(pos.x, TargetX, 1f - Mathf.Exp(-laneChangeLerp * Time.deltaTime));
            transform.position = pos;
        }
    }
}

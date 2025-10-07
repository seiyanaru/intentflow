using System;
using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    public class RunnerController : MonoBehaviour
    {
        [Header("Movement")]
        [SerializeField] private float forwardSpeed = 6f;
        [SerializeField] private float laneWidth = 2f;
        [SerializeField] private float laneChangeLerp = 12f;

        public event Action<Obstacle> Collided;

        public int CurrentLane { get; private set; } = 1;

        public float ForwardSpeed
        {
            get => forwardSpeed;
            set => forwardSpeed = Mathf.Max(0f, value);
        }

        public float LaneWidth => laneWidth;

        private float TargetX => (CurrentLane - 1) * laneWidth;

        public void MoveLeft()
        {
            if (CurrentLane > 0)
            {
                CurrentLane--;
            }
        }

        public void MoveRight()
        {
            if (CurrentLane < 2)
            {
                CurrentLane++;
            }
        }

        private void Update()
        {
            transform.position += Vector3.forward * (forwardSpeed * Time.deltaTime);
            var pos = transform.position;
            pos.x = Mathf.Lerp(pos.x, TargetX, 1f - Mathf.Exp(-laneChangeLerp * Time.deltaTime));
            transform.position = pos;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.TryGetComponent<Obstacle>(out var obstacle))
            {
                Collided?.Invoke(obstacle);
            }
        }
    }
}

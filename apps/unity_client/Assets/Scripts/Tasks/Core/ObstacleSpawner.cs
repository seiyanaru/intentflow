using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    public class ObstacleSpawner : MonoBehaviour
    {
        [SerializeField] private RunnerController runner;
        [SerializeField] private ObjectPool<Obstacle> pool;
        [SerializeField] private float spawnZOffset = 30f;
        [SerializeField] private float spawnInterval = 1.2f;
        [SerializeField] private float lifetime = 10f;
        [SerializeField] private bool activeOnStart = true;

        private float _timer;

        public bool Active { get; set; }

        private void Start()
        {
            if (activeOnStart)
            {
                Active = true;
            }
        }

        public float SpawnInterval
        {
            get => spawnInterval;
            set => spawnInterval = Mathf.Max(0.1f, value);
        }

        public ObjectPool<Obstacle> Pool => pool;

        private void Awake()
        {
            if (!runner)
            {
                runner = FindObjectOfType<RunnerController>();
            }

            if (!pool)
            {
                pool = FindObjectOfType<ObjectPool<Obstacle>>();
            }
        }

        private void Update()
        {
            if (!Active)
            {
                return;
            }

            _timer += Time.deltaTime;
            if (_timer >= spawnInterval)
            {
                _timer = 0f;
                Spawn();
            }
        }

        public void ResetSpawner()
        {
            _timer = 0f;
            pool?.ReleaseAll();
        }

        public void SetPool(ObjectPool<Obstacle> newPool)
        {
            pool = newPool;
        }

        public void Release(Obstacle obstacle)
        {
            pool?.Release(obstacle);
        }

        private void Spawn()
        {
            if (!runner || pool == null)
            {
                return;
            }

            var obstacle = pool.Get();
            if (!obstacle)
            {
                return;
            }

            var lane = Random.Range(0, 3);
            var x = (lane - 1) * runner.LaneWidth;
            var runnerZ = runner.transform.position.z;
            var spawnPos = new Vector3(x, obstacle.transform.position.y, runnerZ + spawnZOffset);
            obstacle.transform.SetParent(transform, false);
            obstacle.transform.SetPositionAndRotation(spawnPos, Quaternion.identity);
            obstacle.Activate(pool, lifetime);
        }
    }
}

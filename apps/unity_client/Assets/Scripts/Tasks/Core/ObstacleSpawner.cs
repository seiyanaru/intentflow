using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    public class ObstacleSpawner : MonoBehaviour
    {
        [SerializeField] private RunnerController runner;
        [SerializeField] private GameObject obstaclePrefab;
        [SerializeField] private float spawnZOffset = 30f;
        [SerializeField] private float spawnInterval = 1.2f;
        [SerializeField] private float lifetime = 10f;

        private float _timer;

        private void Update()
        {
            _timer += Time.deltaTime;
            if (_timer >= spawnInterval)
            {
                _timer = 0f;
                Spawn();
            }
        }

        private void Spawn()
        {
            if (!runner || !obstaclePrefab)
            {
                return;
            }

            var lane = Random.Range(0, 3);
            var x = (lane - 1) * runner.laneWidth;
            var runnerZ = runner.transform.position.z;
            var spawnPos = new Vector3(x, 0f, runnerZ + spawnZOffset);
            var obstacle = Instantiate(obstaclePrefab, spawnPos, Quaternion.identity);
            Destroy(obstacle, lifetime);
        }
    }
}

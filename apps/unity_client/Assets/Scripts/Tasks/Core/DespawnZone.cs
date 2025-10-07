using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    [RequireComponent(typeof(Collider))]
    public class DespawnZone : MonoBehaviour
    {
        [SerializeField] private ObjectPool<Obstacle> pool;

        private void Awake()
        {
            var collider = GetComponent<Collider>();
            collider.isTrigger = true;
            if (!pool)
            {
                pool = FindObjectOfType<ObjectPool<Obstacle>>();
            }
        }

        private void OnTriggerEnter(Collider other)
        {
            if (!pool)
            {
                return;
            }

            if (other.TryGetComponent<Obstacle>(out var obstacle))
            {
                pool.Release(obstacle);
            }
        }

        public void SetPool(ObjectPool<Obstacle> targetPool)
        {
            pool = targetPool;
        }
    }
}

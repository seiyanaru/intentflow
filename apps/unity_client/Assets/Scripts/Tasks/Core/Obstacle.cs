using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    [RequireComponent(typeof(Collider))]
    public class Obstacle : MonoBehaviour
    {
        private ObjectPool<Obstacle> _pool;
        private float _despawnAt;
        private float _lifetime;

        public void Activate(ObjectPool<Obstacle> pool, float lifetime)
        {
            _pool = pool;
            _lifetime = Mathf.Max(0f, lifetime);
            _despawnAt = _lifetime > 0f ? Time.time + _lifetime : float.PositiveInfinity;
            gameObject.SetActive(true);
        }

        public void Release()
        {
            if (_pool != null)
            {
                _pool.Release(this);
            }
            else
            {
                gameObject.SetActive(false);
            }
        }

        private void Update()
        {
            if (Time.time >= _despawnAt)
            {
                Release();
            }
        }
    }
}

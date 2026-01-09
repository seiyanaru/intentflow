using System.Collections.Generic;
using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    public class ObjectPool<T> : MonoBehaviour where T : Component
    {
        [SerializeField] private T prefab;
        [SerializeField] private int initialSize = 8;
        [SerializeField] private bool collectionCheck = true;

        private readonly Queue<T> _available = new Queue<T>();
        private readonly HashSet<T> _active = new HashSet<T>();

        private void Awake()
        {
            if (prefab == null)
            {
                Debug.LogError("ObjectPool prefab is not assigned", this);
                enabled = false;
                return;
            }

            Prewarm(initialSize);
        }

        public void Prewarm(int count)
        {
            for (int i = 0; i < count; i++)
            {
                var instance = CreateInstance();
                _available.Enqueue(instance);
            }
        }

        public T Get()
        {
            if (prefab == null)
            {
                Debug.LogWarning("ObjectPool prefab missing", this);
                return null;
            }

            var item = _available.Count > 0 ? _available.Dequeue() : CreateInstance();
            if (item == null)
            {
                return null;
            }

            _active.Add(item);
            item.gameObject.SetActive(true);
            return item;
        }

        public void Release(T item)
        {
            if (item == null)
            {
                return;
            }

            // Check if already in available queue (double release prevention)
            if (_available.Contains(item))
            {
                return;
            }

            var removed = _active.Remove(item);
            if (!removed && collectionCheck)
            {
                Debug.LogWarning("Attempted to release object that is not tracked by pool", item);
            }

            item.gameObject.SetActive(false);
            item.transform.SetParent(transform, false);
            _available.Enqueue(item);
        }

        public void ReleaseAll()
        {
            foreach (var item in _active)
            {
                item.gameObject.SetActive(false);
                item.transform.SetParent(transform, false);
                _available.Enqueue(item);
            }
            _active.Clear();
        }

        private T CreateInstance()
        {
            var instance = Instantiate(prefab, transform);
            instance.gameObject.SetActive(false);
            return instance;
        }
    }
}

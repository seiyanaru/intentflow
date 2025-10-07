using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    public class TiledGround : MonoBehaviour
    {
        [SerializeField] private RunnerController runner;
        [SerializeField] private Transform[] tiles;
        [SerializeField] private float tileLength = 20f;

        private void Awake()
        {
            if (!runner)
            {
                runner = FindObjectOfType<RunnerController>();
            }
        }

        private void Update()
        {
            if (runner == null || tiles == null || tiles.Length == 0)
            {
                return;
            }

            var runnerZ = runner.transform.position.z;
            var maxZ = GetMaxTileZ();

            for (int i = 0; i < tiles.Length; i++)
            {
                var tile = tiles[i];
                if (!tile)
                {
                    continue;
                }

                if (runnerZ - tile.position.z > tileLength)
                {
                    maxZ += tileLength;
                    var pos = tile.position;
                    pos.z = maxZ;
                    tile.position = pos;
                }
            }
        }

        private float GetMaxTileZ()
        {
            float max = float.MinValue;
            for (int i = 0; i < tiles.Length; i++)
            {
                var tile = tiles[i];
                if (!tile)
                {
                    continue;
                }
                if (tile.position.z > max)
                {
                    max = tile.position.z;
                }
            }

            return max;
        }
    }
}

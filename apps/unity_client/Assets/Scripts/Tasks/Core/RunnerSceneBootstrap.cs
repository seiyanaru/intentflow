using UnityEngine;
using Tasks.Runner3Lane.InputAdapters;
using Tasks.Runner3Lane.UI;

namespace Tasks.Runner3Lane.Core
{
    [DefaultExecutionOrder(-200)]
    public class RunnerSceneBootstrap : MonoBehaviour
    {
        private void Awake()
        {
            var runner = FindObjectOfType<RunnerController>();
            if (!runner)
            {
                Debug.LogWarning("RunnerController not found in scene", this);
            }

            var spawner = FindObjectOfType<ObstacleSpawner>();
            if (!spawner)
            {
                Debug.LogWarning("ObstacleSpawner not found in scene", this);
            }

            var adapter = FindObjectOfType<RunnerInputAdapter>();
            if (!adapter)
            {
                Debug.LogWarning("RunnerInputAdapter not found in scene", this);
            }

            var gm = FindObjectOfType<RunnerGameManager>();
            if (!gm)
            {
                Debug.LogWarning("RunnerGameManager not found in scene", this);
            }

            var hud = FindObjectOfType<Hud>();
            if (!hud)
            {
                Debug.LogWarning("HUD not found in scene", this);
            }
            else if (gm != null)
            {
                hud.Bind(gm);
            }
        }
    }
}

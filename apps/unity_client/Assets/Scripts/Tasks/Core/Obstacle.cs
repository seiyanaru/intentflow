using UnityEngine;

namespace IntentFlow.Tasks.Runner3Lane.Core
{
    [RequireComponent(typeof(Collider))]
    public class Obstacle : MonoBehaviour
    {
        private void OnTriggerEnter(Collider other)
        {
            var gm = FindObjectOfType<RunnerGameManager>();
            if (gm)
            {
                gm.OnHitObstacle();
            }
            Destroy(gameObject);
        }
    }
}

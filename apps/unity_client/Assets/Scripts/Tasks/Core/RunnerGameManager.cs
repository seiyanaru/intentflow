using UnityEngine;

namespace IntentFlow.Tasks.Runner3Lane.Core
{
    public class RunnerGameManager : MonoBehaviour
    {
        [SerializeField] private RunnerController runner;
        [SerializeField] private int lives = 3;

        public int Score { get; private set; }
        public int Lives => lives;

        private void Awake()
        {
            if (!runner)
            {
                runner = FindObjectOfType<RunnerController>();
            }
        }

        private void Update()
        {
            if (runner)
            {
                Score = Mathf.RoundToInt(runner.transform.position.z);
            }
        }

        public void OnHitObstacle()
        {
            lives--;
            if (lives <= 0)
            {
                Time.timeScale = 0f;
                Debug.Log("Game Over");
            }
        }
    }
}

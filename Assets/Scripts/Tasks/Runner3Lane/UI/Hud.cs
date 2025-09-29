using UnityEngine;
using UnityEngine.UI;
using IntentFlow.Tasks.Runner3Lane.Core;
using IntentFlow.Tasks.Runner3Lane.InputAdapters;

namespace IntentFlow.Tasks.Runner3Lane.UI
{
    public class Hud : MonoBehaviour
    {
        [SerializeField] private Text scoreText;
        [SerializeField] private Text livesText;
        [SerializeField] private Text successText;

        [SerializeField] private RunnerGameManager gameManager;
        [SerializeField] private RunnerInputAdapter inputAdapter;

        private void Awake()
        {
            if (!gameManager)
            {
                gameManager = FindObjectOfType<RunnerGameManager>();
            }
            if (!inputAdapter)
            {
                inputAdapter = FindObjectOfType<RunnerInputAdapter>();
            }
        }

        private void Update()
        {
            if (!gameManager)
            {
                gameManager = FindObjectOfType<RunnerGameManager>();
            }
            if (!inputAdapter)
            {
                inputAdapter = FindObjectOfType<RunnerInputAdapter>();
            }

            if (scoreText && gameManager)
            {
                scoreText.text = $"Score: {gameManager.Score}";
            }
            if (livesText && gameManager)
            {
                livesText.text = $"Lives: {gameManager.Lives}";
            }
            if (successText && inputAdapter)
            {
                successText.text = $"Ops: {inputAdapter.SuccessCount}";
            }
        }
    }
}

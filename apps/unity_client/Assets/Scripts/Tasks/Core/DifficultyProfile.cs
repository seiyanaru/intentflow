using UnityEngine;

namespace Tasks.Runner3Lane.Core
{
    [CreateAssetMenu(menuName = "Runner/DifficultyProfile")]
    public class DifficultyProfile : ScriptableObject
    {
        public AnimationCurve spawnIntervalOverTime = AnimationCurve.Constant(0f, 60f, 1.2f);
        public AnimationCurve forwardSpeedOverTime = AnimationCurve.Linear(0f, 6f, 60f, 10f);
        public float minSpawnInterval = 0.4f;
        public float maxForwardSpeed = 12f;
    }
}

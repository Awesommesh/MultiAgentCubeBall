using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct ForwardModelJob : IJobParallelFor {
    public NeuralNetwork model;
    [ReadOnly]
    public NativeArray<NDArray> states;
    [ReadOnly]
    NDArray log_std;
    [WriteOnly]
    public NativeArray<NDArray> actions;
    [WriteOnly]
    public NativeArray<NDArray> log_probs;
    public void Execute(int i) {
        NDArray dist = model.Forward(states[i]);
        NDArray nextAction = actions[i];
        NDArray nextLogProb = log_probs[i];
        //ASSUMING MODEL OUTPUTS MEAN AND STD IS FIXED
        for (int j = 0; j < dist[i]; j++) {
            nextAction[j] = GaussianDistribution.NextGaussian(dist[j], log_std[j]);
            nextLogProb[j] = GaussianDistribution.log_prob(nextAction[j], dist[j], log_std[j]);
        }
    }
}

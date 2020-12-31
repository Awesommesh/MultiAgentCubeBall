using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct ActorForwardModelJob : IJob {
    public NeuralNetwork model;
    [ReadOnly]
    public NDArray state;
    [WriteOnly]
    public NDArray action;
    [WriteOnly]
    public NDArray log_prob;
    public void Execute() {
        NDArray dist = model.Forward(state);
        //ASSUMING MODEL OUTPUTS MEAN AND STD IS FIXED
        for (int j = 0; j < dist.numElements; j++) {
            action[j] = GaussianDistribution.NextGaussian(dist[j], model.log_std[j]);
            log_prob[j] = GaussianDistribution.log_prob(action[j], dist[j], model.log_std[j]);
        }
    }
}

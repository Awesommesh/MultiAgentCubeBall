using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct ForwardModelJob : IJobParallelFor {
    public NeuralNetwork model;
    [ReadOnly]
    public NativeArray<NDArray> states;
    [WriteOnly]
    public NativeArray<NDArray> actions;
    public void Execute(int i) {
        NDArray dist = model.Forward(states[i]);
        NDArray nextAction = actions[i];
        //ASSUMING MODEL OUTPUTS 2 VALUES, MEAN AND STD. !!!NOTE: OUTPUT OF NETWORK MUST BE EVEN AS A RESULT!!!
        for (int j = 0; j < dist[i]; j+=2) {
            nextAction[j/2] = GaussianDistribution.NextGaussian(dist[j], dist[j+1]);
        }
    }
}

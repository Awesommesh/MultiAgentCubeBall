using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct CriticForwardModelJob : IJob {
    public NeuralNetwork model;
    [ReadOnly]
    public NDArray state;
    [WriteOnly]
    public double value;
    public void Execute() {
        NDArray valueOut = model.Forward(state);
        value = valueOut[0];
    }
}

using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct CalculateGAEJob : IJobParallelFor {
    [WriteOnly]
    NativeArray<NDArray> returns;
    [ReadOnly]
    NativeArray<NDArray> rewards;
    [ReadOnly]
    NativeArray<NDArray> values;
    [ReadOnly]
    NativeArray<NDArray> mask;
    [ReadOnly]
    int gamma;
    [ReadOnly]
    int lambda;
    [ReadOnly]
    double next_value;
    [ReadOnly]
    int numSteps;
    
    public void Execute(int i) {
        double gae = rewards[i][numSteps-1] + gamma * next_value * mask[i][numSteps - 1] - values[i][numSteps- 1];
        returns[i].set(0, gae + values[i][numSteps - 1]);
        for (int j = numSteps-2; j >= 0; j--) {
            double delta = rewards[i][j] + gamma * values[i][j+1] * mask[i][j] - values[i][j];
            gae = delta + gamma * lambda * mask[i][j] * gae;
            returns[i].set(numSteps-j-1, gae + values[i][j]);
        }
    }
}

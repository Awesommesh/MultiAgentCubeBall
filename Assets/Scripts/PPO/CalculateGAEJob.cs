using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct CalculateGAEJob : IJobParallelFor {
    [WriteOnly]
    NativeArray<NDArray> returns;
    [WriteOnly]
    NativeArray<NDArray> advantages;
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
        double mean = 0;
        double std = 0;
        double gae = rewards[i][numSteps-1] + gamma * next_value * mask[i][numSteps - 1] - values[i][numSteps- 1];
        advantages[i].set(0, gae);
        mean += gae;
        returns[i].set(0, gae + values[i][numSteps - 1]);
        for (int j = numSteps-2; j >= 0; j--) {
            double delta = rewards[i][j] + gamma * values[i][j+1] * mask[i][j] - values[i][j];
            gae = delta + gamma * lambda * mask[i][j] * gae;
            advantages[i].set(numSteps-j-1, gae);
            mean += gae;
            returns[i].set(numSteps-j-1, gae + values[i][j]);
        }
        mean /= numSteps;
        for (int j = 0; j < numSteps; j++) {
            std += math.pow(math.abs(advantages[i][j] - mean), 2);
        }
        std /= numSteps;
        std = math.sqrt(std);
        for (int j = 0; j < numSteps; j++) {
            advantages[i].set(j, ((advantages[i][j] - mean)/std));
        }
    }
}

/*using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct CalculateGAEJob : IJob {
    [WriteOnly]
    NDArray returns;
    [WriteOnly]
    NDArray advantages;
    [ReadOnly]
    NDArray rewards;
    [ReadOnly]
    NDArray values;
    [ReadOnly]
    NDArray mask;
    [ReadOnly]
    int gamma;
    [ReadOnly]
    int lambda;
    [ReadOnly]
    double next_value;
    [ReadOnly]
    int numSteps;
    
    public void Execute() {
        double mean = 0;
        double std = 0;
        double gae = rewards[numSteps-1] + gamma * next_value * mask[numSteps - 1] - values[numSteps- 1];
        advantages.set(0, gae);
        mean += gae;
        returns.set(0, gae + values[numSteps - 1]);
        for (int i = numSteps-2; i >= 0; i--) {
            double delta = rewards[i] + gamma * values[i+1] * mask[i] - values[i];
            gae = delta + gamma * lambda * mask[i] * gae;
            advantages.set(numSteps-i-1, gae);
            mean += gae;
            returns.set(numSteps-i-1, gae + values[i]);
        }
        mean /= numSteps;
        for (int i = 0; i < numSteps; i++) {
            std += math.pow(math.abs(advantages[i] - mean), 2);
        }
        std /= numSteps;
        std = math.sqrt(std);
        for (int i = 0; i < numSteps; i++) {
            advantages.set(i, ((advantages[i] - mean)/std));
        }
    }
}
*/
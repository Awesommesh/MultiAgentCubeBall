using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;

//Calculates returns and advantages for 1 agents experiences
[BurstCompile]
public struct CalculateGAEJob : IJob {
    [ReadOnly]
    public NativeArray<double> rewards;
    [ReadOnly]
    public NativeArray<double> values;
    [ReadOnly]
    public NativeArray<int> mask;
    [ReadOnly]
    public double gamma;
    [ReadOnly]
    public double lambda;
    [ReadOnly]
    public double next_value;
    [ReadOnly]
    public int numSteps;
    [ReadOnly]
    public int agentInd;
    [ReadOnly]
    public int TEAM_SIZE;

    [WriteOnly]
    public NativeArray<double> returns;
    [WriteOnly]
    public NativeArray<double> advantages;
    
    public void Execute() {
        double mean = 0;
        double std = 0;
        double gae = rewards[numSteps-1] + gamma * next_value * mask[numSteps - 1] - values[numSteps- 1];
        setAdvantages(0, gae);
        mean += gae;
        setReturns(0, gae + values[numSteps - 1]);
        for (int i = numSteps-2; i >= 0; i--) {
            double delta = rewards[i] + gamma * values[i+1] * mask[i] - values[i];
            gae = delta + gamma * lambda * mask[i] * gae;
            setAdvantages(numSteps-i-1, gae);
            mean += gae;
            setReturns(numSteps-i-1, gae + values[i]);
        }
        mean /= numSteps;
        for (int i = 0; i < numSteps; i++) {
            std += math.pow(math.abs(getAdvantages(i) - mean), 2);
        }
        std /= numSteps;
        std = math.sqrt(std);
        for (int i = 0; i < numSteps; i++) {
            setAdvantages(i, ((getAdvantages(i) - mean)/std));
        }
    }

    public void setAdvantages(int index, double value) {
        advantages[TEAM_SIZE*index + agentInd] = value;
    }

    public double getAdvantages(int index) {
        return advantages[TEAM_SIZE*index + agentInd];
    }

    public void setReturns(int index, double value) {
        returns[TEAM_SIZE*index + agentInd] = value;
    }
}
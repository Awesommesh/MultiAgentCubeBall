using UnityEngine.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct PPOUpdateJob : IJobParallelFor {
    [ReadOnly]
    NativeArray<NDArray> minibatches;
    [ReadOnly]
    NativeArray<NDArray> states;
    [ReadOnly]
    NativeArray<NDArray> action;
    [ReadOnly]
    NativeArray<NDArray> old_log_probs;
    [ReadOnly]
    NativeArray<NDArray> returns;
    [ReadOnly]
    NativeArray<NDArray> advantage;
    [ReadOnly]
    int PPO_EPILSON;
    [ReadOnly]
    int CRITIC_DISCOUNT;
    [ReadOnly]
    int ENTROPY_BETA;
    NeuralNetwork actor;
    NeuralNetwork critic;
    
    public void Execute(int i) {
        for (int j = 0; j < minibatches[i].numElements; j++) {
            int index = (int)minibatches[i][j];
            NDArray dist = actor.Forward(states[index]);
            NDArray value = critic.Forward(states[index]);
            double entropy = GaussianDistribution.entropy(actor.log_std.Mean());
            NDArray new_log_probs = GaussianDistribution.log_prob(action[index], dist, actor.log_std, Allocator.TempJob);
            NDArray ratio = NDArray.Exp(new_log_probs - old_log_probs[index]);
            NDArray surr1 = ratio * advantage[index];
            NDArray surr2 = NDArray.Clamp(ratio, 1-PPO_EPILSON, 1+PPO_EPILSON) * advantage[index];
            double actor_loss = -(NDArray.Min(surr1, surr2)).Mean();
            double critic_loss = (NDArray.Pow(returns[index]-value[index], 2)).Mean();
            double total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy;

            //Actor NN BackProp
            NDArray minBacksurr1 = -(1/surr1.numElements) * (surr1 < surr2) * surr1;
            NDArray minBacksurr2 = -(1/surr1.numElements) * (surr2 < surr1) * NDArray.Clamp_Back(ratio, 1-PPO_EPILSON, 1+PPO_EPILSON) * surr1;
            NDArray minBackDist = (minBacksurr1 + minBacksurr2) * GaussianDistribution.log_prob_back(action[index], dist, actor.log_std);
            actor.Backward(minBackDist);

            //Critic NN BackProp
            NDArray backValue = -CRITIC_DISCOUNT*(2/returns[index].numElements)*(returns[index] - value[index]);
            critic.Backward(backValue);
            
            new_log_probs.Dispose();
        }
    }
}
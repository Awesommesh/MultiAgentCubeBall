using UnityEngine.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct PPOUpdateJob : IJob {
    [ReadOnly]
    public NativeArray<int> minibatches;
    [ReadOnly]
    public int batchInd;
    [ReadOnly]
    public NativeArray<double> actionDists;
    [ReadOnly]
    public NativeArray<double> stateVals;
    [ReadOnly]
    public NativeArray<double> actions;
    [ReadOnly]
    public NativeArray<double> old_log_probs;
    [ReadOnly]
    public NativeArray<double> returns;
    [ReadOnly]
    public NativeArray<double> advantages;
    [ReadOnly]
    public NativeArray<double> log_std;
    [ReadOnly]
    public double log_std_mean;
    [ReadOnly]
    public int NUM_ACTIONS;
    [ReadOnly]
    public double PPO_EPILSON;
    [ReadOnly]
    public double CRITIC_DISCOUNT;
    [ReadOnly]
    public double ENTROPY_BETA;
    [ReadOnly]
    public int MINI_BATCH_SIZE;

    [WriteOnly] 
    public NativeArray<double> actorGrad;
    [WriteOnly]
    public NativeArray<double> criticGrad;

    public void Execute() {
        int minibatch_ind = batchInd*MINI_BATCH_SIZE;
        double actor_loss = 0;
        double critic_loss = 0;
        double entropy = GaussianDistribution.entropy(log_std_mean);
        for (int j = minibatch_ind; j < minibatch_ind + MINI_BATCH_SIZE; j++) {
            int index = minibatches[j];
            int actionIndex = index*NUM_ACTIONS;
            NativeArray<double> new_log_probs = GaussianDistribution.log_prob(actions, actionDists, log_std, actionIndex, NUM_ACTIONS, Allocator.Temp);
            NativeArray<double> log_prob_back = GaussianDistribution.log_prob_back(actions, actionDists, log_std, actionIndex, NUM_ACTIONS, Allocator.Temp);
            for (int k = 0; k < NUM_ACTIONS; k++) {
                double ratio = math.exp(new_log_probs[k] - old_log_probs[actionIndex + k]);
                double surr1 = ratio * advantages[index];
                double surr2 = math.clamp(ratio, 1-PPO_EPILSON, 1+PPO_EPILSON) * advantages[index];
                actor_loss += math.min(surr1, surr2);

                //Actor Grad
                int surr1Less = surr1 < surr2 ? 1 : 0;
                int surr2Less = surr2 < surr1 ? 1 : 0;
                int clamp_back = (ratio > 1+PPO_EPILSON || ratio < 1-PPO_EPILSON) ? 0 : 1;
                actorGrad[(j-minibatch_ind)*NUM_ACTIONS + k] = (-1/MINI_BATCH_SIZE*NUM_ACTIONS)*(surr1Less * surr1 + surr2Less * clamp_back * surr1) * log_prob_back[k];
            }
            
            critic_loss += math.pow(returns[j] - stateVals[j], 2);

            //Critic Grad
            criticGrad[j-minibatch_ind] = -CRITIC_DISCOUNT * (2 / MINI_BATCH_SIZE) * (returns[j-minibatch_ind] - stateVals[j-minibatch_ind]);
        }
        actor_loss /= MINI_BATCH_SIZE*NUM_ACTIONS;
        actor_loss *= -1;
        critic_loss /= MINI_BATCH_SIZE;
        double total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy;
    }
}
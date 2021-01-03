using UnityEngine.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Jobs;

[BurstCompile]
public struct PPOUpdateJob : IJob {
    [ReadOnly]
    public NativeArray<double> actionDists;
    [ReadOnly]
    public double stateVal;
    [ReadOnly]
    public NativeArray<double> actions;
    [ReadOnly]
    public NativeArray<double> old_log_probs;
    [ReadOnly]
    public double returns;
    [ReadOnly]
    public double advantage;
    [ReadOnly]
    public NativeArray<double> log_stds;
    [ReadOnly]
    public double entropy;
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
    public double criticGrad;

    [ReadOnly]
    public double actor_loss;
    [ReadOnly]
    public double critic_loss;

    [WriteOnly]
    public double nextActor_loss;
    [WriteOnly]
    public double nextCritic_loss;

    public void Execute() {
        NativeArray<double> new_log_probs = GaussianDistribution.log_prob(actions, actionDists, log_stds, Allocator.Temp);
        NativeArray<double> log_prob_back = GaussianDistribution.log_prob_back(actions, actionDists, log_stds, Allocator.Temp);
        for (int i = 0; i < NUM_ACTIONS; i++) {
            double ratio = math.exp(new_log_probs[i] - old_log_probs[i]);
            double surr1 = ratio * advantage;
            double surr2 = math.clamp(ratio, 1-PPO_EPILSON, 1+PPO_EPILSON) * advantage;
            nextActor_loss = actor_loss + math.min(surr1, surr2);

            //Actor Grad
            int surr1Less = surr1 < surr2 ? 1 : 0;
            int surr2Less = surr2 < surr1 ? 1 : 0;
            int clamp_back = (ratio > 1+PPO_EPILSON || ratio < 1-PPO_EPILSON) ? 0 : 1;
            actorGrad[i] = (-1/MINI_BATCH_SIZE*NUM_ACTIONS)*(surr1Less * surr1 + surr2Less * clamp_back * surr1) * log_prob_back[i];
        }

        nextCritic_loss = critic_loss + math.pow(returns - stateVal, 2);

        //Critic Grad
        criticGrad = -CRITIC_DISCOUNT * (2 / MINI_BATCH_SIZE) * (returns - stateVal);
        nextCritic_loss = 2;
    }
}
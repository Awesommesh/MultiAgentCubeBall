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
    public NativeArray<double> stateVal;
    [ReadOnly]
    public NativeArray<double> actions;
    [ReadOnly]
    public NativeArray<double> old_log_probs;
    [ReadOnly]
    public NativeArray<double> returns;
    [ReadOnly]
    public NativeArray<double> advantage;
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
    public NativeArray<double> criticGrad;
    [WriteOnly]
    public NativeArray<double> actor_loss;
    [WriteOnly]
    public NativeArray<double> critic_loss;

    public void Execute() {
        NativeArray<double> new_log_probs = GaussianDistribution.log_prob(actions, actionDists, log_stds, NUM_ACTIONS, MINI_BATCH_SIZE, Allocator.Temp);
        NativeArray<double> log_prob_back = GaussianDistribution.log_prob_back(actions, actionDists, log_stds, NUM_ACTIONS, MINI_BATCH_SIZE, Allocator.Temp);
        double local_AL = 0;
        double local_CL = 0;
        for (int j = 0; j < MINI_BATCH_SIZE; j++) {
            for (int i = 0; i < NUM_ACTIONS; i++) {
                double ratio = math.exp(new_log_probs[i*NUM_ACTIONS+j]-old_log_probs[i*NUM_ACTIONS+j]);
                double surr1 = ratio * advantage[i];
                double surr2 = math.clamp(ratio, 1-PPO_EPILSON, 1+PPO_EPILSON) * advantage[i];
                local_AL += math.min(surr1, surr2);
                
                //Actor Grad
                int surr1Less = surr1 <= surr2 ? 1 : 0;
                int clamp_back = (ratio > 1+PPO_EPILSON || ratio < 1-PPO_EPILSON) ? 0 : 1;
                actorGrad[i*MINI_BATCH_SIZE+j] = (-1*((double)1/(MINI_BATCH_SIZE*NUM_ACTIONS)) * (surr1Less + (1 - surr1Less)*clamp_back)
                    * ratio * advantage[i] * log_prob_back[i*NUM_ACTIONS + j]);
            }
            local_CL += math.pow(returns[j] - stateVal[j], 2);

            //Critic Grad
            criticGrad[j] = -(CRITIC_DISCOUNT*2/MINI_BATCH_SIZE)*(returns[j] - stateVal[j]);
        }
        actor_loss[0] = local_AL;
        critic_loss[0] = local_CL;
        new_log_probs.Dispose();
        log_prob_back.Dispose();
    }
}
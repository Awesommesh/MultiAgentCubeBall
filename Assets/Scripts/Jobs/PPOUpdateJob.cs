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
    public NativeArray<double> stds;
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
    //[WriteOnly]
    //public NativeArray<double> entropy_loss;

    public void Execute() {
        NativeArray<double> new_log_probs = GaussianDistribution.log_prob(actions, actionDists, stds, NUM_ACTIONS, MINI_BATCH_SIZE, Allocator.Temp);
        NativeArray<double> log_prob_back = GaussianDistribution.log_prob_back(actions, actionDists, stds, NUM_ACTIONS, MINI_BATCH_SIZE, Allocator.Temp);
        double local_AL = 0;
        double local_CL = 0;
        double actorGradConst = -((double)1/(MINI_BATCH_SIZE*NUM_ACTIONS));
        double criticGradConst =  -(CRITIC_DISCOUNT*2/MINI_BATCH_SIZE);
        //double local_EL = 0;
        
        for (int i = 0; i < NUM_ACTIONS; i++) {
            int rowInd = i*MINI_BATCH_SIZE;
            for (int j = 0; j < MINI_BATCH_SIZE; j++) {
                double ratio = math.exp(new_log_probs[rowInd+j]-old_log_probs[rowInd+j]);
                double surr1 = ratio * advantage[j];
                double surr2 = math.clamp(ratio, 1-PPO_EPILSON, 1+PPO_EPILSON) * advantage[j];
                local_AL -= math.min(surr1, surr2);
                
                //Actor Grad
                int surr1Less = surr1 <= surr2 ? 1 : 0;
                int clamp_back = (ratio > 1+PPO_EPILSON || ratio < 1-PPO_EPILSON) ? 0 : 1;
                actorGrad[rowInd+j] = actorGradConst * (surr1Less + (1 - surr1Less)*clamp_back)
                    * ratio * advantage[j] * log_prob_back[rowInd + j];

                //local_EL += 
            }
            
        }
        for (int i = 0; i < MINI_BATCH_SIZE; i++) {
            local_CL += math.pow(returns[i] - stateVal[i], 2);

            //Critic Grad
            criticGrad[i] = criticGradConst*(returns[i] - stateVal[i]);
        }
        actor_loss[0] = local_AL;
        critic_loss[0] = local_CL;
        new_log_probs.Dispose();
        log_prob_back.Dispose();
    }
}
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
    NativeArray<NDArray> advantage; // = returns - value (NORMALIZE PLS)
    NeuralNetwork actor;
    NeuralNetwork critic;
    
    public void Execute(int i) {
        for (int j = 0; j < minibatches[i].numElements; j++) {
            int index = (int)minibatches[i][j];
            NDArray dist = actor.Forward(states[index]);
            NDArray value = critic.Forward(states[index]);
            
        }
    }
}
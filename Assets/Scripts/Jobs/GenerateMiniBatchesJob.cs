using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;
[BurstCompile]
public struct Generateminibatch : IJob {
    [WriteOnly]
    NativeArray<int> minibatch;
    [ReadOnly]
    uint seed;
    //An array of lenght batch_size with each index containing a unique index from 0 - batch_size-1
    NativeArray<int> shuffle;

    public void Execute() {
        Random sampler = new Random(seed);
        int minibatch_size = minibatch.Length;
        for (int i = shuffle.Length-1; i > shuffle.Length-1-minibatch_size; i--) {
            int randIndex = (int)sampler.NextDouble()*(i+1);
            int temp =  shuffle[randIndex];
            shuffle[randIndex] = shuffle[i];
            shuffle[i] = temp;
            minibatch[shuffle.Length-1-i] = temp;
        }
    }
}

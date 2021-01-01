using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;
[BurstCompile]
public struct GenerateMiniBatchesJob : IJobParallelFor {
    [ReadOnly]
    public uint seed;
    [ReadOnly]
    public int MINI_BATCH_SIZE;

    [WriteOnly]
    public NativeArray<int> minibatches;

    //An array of length batch_size with each index containing a unique index from 0 - batch_size-1
    public NativeArray<int> shuffle;

    public void Execute(int i) {
        Random sampler = new Random(seed);
        int startInd = i*MINI_BATCH_SIZE;
        for (int j = shuffle.Length-1; j > shuffle.Length-1-MINI_BATCH_SIZE; j--) {
            int randIndex = (int)sampler.NextDouble()*(j+1);
            int temp =  shuffle[randIndex];
            shuffle[randIndex] = shuffle[j];
            shuffle[j] = temp;
            minibatches[startInd+shuffle.Length-1-i] = temp;
        }
    }
}

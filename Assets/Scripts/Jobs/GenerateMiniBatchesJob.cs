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
    [ReadOnly]
    public int BATCH_SIZE;


    [WriteOnly]
    [NativeDisableParallelForRestriction]
    public NativeArray<int> minibatches;

    

    public void Execute(int i) {
        Random sampler = new Random(seed);

        //An array of length batch_size with each index containing a unique index from 0 - batch_size-1
        NativeArray<int>shuffle = new NativeArray<int>(BATCH_SIZE, Allocator.Temp);
        for (int j = 0; j < shuffle.Length; j++) {
            shuffle[j] = j;
        }

        int startInd = i*MINI_BATCH_SIZE;
        for (int j = shuffle.Length-1; j > shuffle.Length-1-MINI_BATCH_SIZE; j--) {
            int randIndex = (int)sampler.NextDouble()*j;
            int temp =  shuffle[randIndex];
            shuffle[randIndex] = shuffle[j];
            shuffle[j] = temp;
            minibatches[startInd+shuffle.Length-1-j] = temp;
        }
        shuffle.Dispose();
    }
}

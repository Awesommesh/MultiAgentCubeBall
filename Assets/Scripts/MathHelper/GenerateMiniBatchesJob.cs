using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;
[BurstCompile]
public struct GenerateMiniBatches : IJobParallelFor {
    [WriteOnly]
    NativeArray<NDArray> minibatches;
    [ReadOnly]
    uint seed;
    //An array of lenght batch_size with each index containing a unique index from 0 - batch_size-1
    NativeArray<int> shuffle;

    public void Execute(int i ) {
        Random sampler = new Random(seed);
        int minibatch_size = minibatches[i].numElements;
        for (int j = shuffle.Length-1; j > shuffle.Length-1-minibatch_size; j--) {
            int randIndex = (int)sampler.NextDouble()*(j+1);
            int temp =  shuffle[randIndex];
            shuffle[randIndex] = shuffle[j];
            shuffle[j] = temp;
            minibatches[i].set(shuffle.Length-1-j, temp);
        }
    }
}

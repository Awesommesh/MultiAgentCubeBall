using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;

[BurstCompile]
public struct DotProductJob : IJobParallelFor {
    [ReadOnly]
    public NativeArray<double> a;
    [ReadOnly]
    public NativeArray<double> b;
    [ReadOnly]
    public int len;
    [ReadOnly]
    public int rowInd;
    [ReadOnly]
    public int outputLen;

    [WriteOnly]
    [NativeDisableParallelForRestriction]
    public NativeArray<double> output;

    public void Execute(int i) {
        double dotprod = 0;
        for (int j = 0; j < len; j++) {
            dotprod += a[rowInd*len + j]*b[j*len+i];
        }
        output[rowInd*outputLen + i] = dotprod;
    }
}
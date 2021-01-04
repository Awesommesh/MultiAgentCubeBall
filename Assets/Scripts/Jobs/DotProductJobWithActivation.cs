using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

[BurstCompile]
public struct DotProductJobWithActivations : IJobParallelFor {
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
    [ReadOnly]
    public ActivationType activation;

    [WriteOnly]
    [NativeDisableContainerSafetyRestriction]
    public NativeArray<double> output;
    [WriteOnly]
    [NativeDisableContainerSafetyRestriction]
    public NativeArray<double> activationOutput;

    public void Execute(int i) {
        double dotprod = 0;
        for (int j = 0; j < len; j++) {
            dotprod += a[rowInd*len + j]*b[j*outputLen+i];
        }
        output[rowInd*outputLen + i] = dotprod;
        activationOutput[rowInd*outputLen + i] = NativeNDOps.ActivationFunction(activation, dotprod);
    }
}
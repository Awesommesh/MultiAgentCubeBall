using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;

[BurstCompile]
public struct NNStepForwardJob : IJob {
    [ReadOnly]
    public NativeArray<double> weights;
    [ReadOnly]
    public NativeArray<int> weightsShape;
    [ReadOnly]
    public NativeArray<double> input;
    [ReadOnly]
    public ActivationType activation;
    
    [WriteOnly]
    public NativeArray<double> layerOutput;
    [WriteOnly]
    public NativeArray<double> activationOutput;

    public void Execute() {
        NativeArray<int> inputShape = new NativeArray<int>(2, Allocator.TempJob);
        inputShape[0] = input.Length;
        inputShape[1] = 1;
        NativeNDOps.Dot(weights, weightsShape, input, inputShape, layerOutput);
        inputShape.Dispose();
        NativeNDOps.ActivationFunction(activation, layerOutput, activationOutput);
    }

}
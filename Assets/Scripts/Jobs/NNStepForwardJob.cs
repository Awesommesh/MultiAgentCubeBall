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
    public int numInputs;
    [ReadOnly]
    public ActivationType activation;
    
    public NativeArray<double> layerOutput;
    [WriteOnly]
    public NativeArray<double> activationOutput;

    public void Execute() {
        NativeArray<int> inputShape = new NativeArray<int>(2, Allocator.Temp);
        inputShape[0] = weightsShape[1];
        inputShape[1] = numInputs;
        NativeNDOps.Dot(weights, weightsShape, 0, input, inputShape, 0, layerOutput);
        inputShape.Dispose();
        NativeNDOps.ActivationFunction(activation, layerOutput, activationOutput);
    }

}
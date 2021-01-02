using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;

[BurstCompile]
public struct NNStepBackwardJob : IJob {
    [ReadOnly]
    public NativeArray<double> weights;
    [ReadOnly]
    public NativeArray<int> weightsShape;
    [ReadOnly]
    public NativeArray<double> layerInput;
    [ReadOnly]
    public NativeArray<double> activationInput;
    [ReadOnly]
    public NativeArray<double> grad;
    [ReadOnly]
    public ActivationType activation;

    public NativeArray<double> weightsGrad;
    public NativeArray<double> layerGrad;

    public void Execute() {
        //Activation Gradient
        NativeArray<double> activtionGrad = new NativeArray<double>(grad.Length, Allocator.Temp);
        NativeArray<int> activationGradShape =  new NativeArray<int>(2, Allocator.Temp);
        activationGradShape[0] = activationInput.Length;
        activationGradShape[1] = 1;
        NativeNDOps.ActivationFunctionBack(activation, activationInput, grad, activtionGrad);
        
        //Weights Gradient = activationGrad.Dot(layerInput.T());
        NativeArray<int> layerInputShape =  new NativeArray<int>(2, Allocator.Temp);
        layerInputShape[0] = 1;
        layerInputShape[1] = layerInput.Length;
        NativeNDOps.Dot(activtionGrad, activationGradShape, layerInput, layerInputShape, weightsGrad);

        //Layer Gradient = weights.T().Dot(activationGrad)
        NativeArray<double> weightsTranspose = new NativeArray<double>(weights.Length, Allocator.Temp);
        NativeArray<int> weightsTransposeShape = new NativeArray<int>(2, Allocator.Temp);
        weightsTransposeShape[0] = weightsShape[1];
        weightsTransposeShape[1] = weightsShape[0];
        NativeNDOps.Transpose(weights, weightsShape, weightsTranspose);
        
        NativeNDOps.Dot(weightsTranspose, weightsTransposeShape, activtionGrad, activationGradShape, layerGrad);

        //Dispose shape arrays and activation gradients
        activtionGrad.Dispose();
        activationGradShape.Dispose();
        layerInputShape.Dispose();
        weightsTranspose.Dispose();
        weightsTransposeShape.Dispose();
    }
}
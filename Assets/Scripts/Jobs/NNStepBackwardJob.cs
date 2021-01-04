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
    public int numGrads;
    [ReadOnly]
    public ActivationType activation;

    public NativeArray<double> weightsGrad;
    public NativeArray<double> layerGrad;

    public void Execute() {
        //Activation Gradient
        NativeArray<double> activationGrad = new NativeArray<double>(grad.Length, Allocator.Temp);
        NativeNDOps.ActivationFunctionBack(activation, activationInput, grad, activationGrad);
        
        //Weights Gradient = activationGrad.Dot(layerInput.T());
        NativeArray<double> inputTranspose = new NativeArray<double>(weightsShape[1]*numGrads, Allocator.Temp);
        NativeArray<int> layerInputShape =  new NativeArray<int>(2, Allocator.Temp);
        layerInputShape[0] = weightsShape[1];
        layerInputShape[1] = numGrads;
        NativeNDOps.Transpose(layerInput, layerInputShape, inputTranspose);

        NativeArray<int> inputTransposeShape = new NativeArray<int>(2, Allocator.Temp);
        inputTransposeShape[0] = numGrads;
        inputTransposeShape[1] = weightsShape[1];
        NativeArray<int> gradShape = new NativeArray<int>(2, Allocator.Temp);
        gradShape[0] = weightsShape[0];
        gradShape[1] = numGrads;
        NativeNDOps.Dot(activationGrad, gradShape, 0, inputTranspose, inputTransposeShape, 1, weightsGrad);
        
        //Layer Gradient = weights.T().Dot(activationGrad)
        NativeArray<double> weightsTranspose = new NativeArray<double>(weights.Length, Allocator.Temp);
        NativeArray<int> weightsTransposeShape = new NativeArray<int>(2, Allocator.Temp);
        weightsTransposeShape[0] = weightsShape[1];
        weightsTransposeShape[1] = weightsShape[0];
        NativeNDOps.Transpose(weights, weightsShape, weightsTranspose);
        
        NativeNDOps.Dot(weightsTranspose, weightsTransposeShape, 1, activationGrad, gradShape, 0, layerGrad);

        //Dispose shape arrays and activation gradientsd
        gradShape.Dispose();
        activationGrad.Dispose();
        inputTranspose.Dispose();
        layerInputShape.Dispose();
        inputTransposeShape.Dispose();
        weightsTranspose.Dispose();
        weightsTransposeShape.Dispose();
    }
}
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;

//[BurstCompile]
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
        
        NativeArray<int> layerInputShape =  new NativeArray<int>(2, Allocator.Temp);
        layerInputShape[0] = weightsShape[1];
        layerInputShape[1] = numGrads;
 
        NativeArray<int> gradShape = new NativeArray<int>(2, Allocator.Temp);
        gradShape[0] = weightsShape[0];
        gradShape[1] = numGrads;
        NativeNDOps.Dot(activationGrad, gradShape, 0, layerInput, layerInputShape, 1, weightsGrad);
        
        NativeArray<double> allLayerGrads = new NativeArray<double>(layerInput.Length, Allocator.Temp);
        NativeNDOps.Dot(weights, weightsShape, 1, activationGrad, gradShape, 0, allLayerGrads);
        NativeNDOps.CopyPartial(allLayerGrads, layerGrad, layerGrad.Length);
        allLayerGrads.Dispose();

        //Dispose shape arrays and activation gradientsd
        gradShape.Dispose();
        activationGrad.Dispose();
        layerInputShape.Dispose();
    }
}
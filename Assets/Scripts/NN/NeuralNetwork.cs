using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
public struct NeuralNetwork { //Change to call small jobs to do all calcs
    NDArray[] weights;
    //Adam Optimization Parameters
    private NDArray[] V_dw;
    private NDArray[] S_dw;
    private double alpha;
    private double beta1;
    private double beta2;
    private double epsilon;
    private int iteration;
    int numLayers;
    ActivationType[] activations;
    int numInputs;
    int numOutputs;
    public NativeArray<double> log_std;
    public double log_std_mean;
    NDArray[] inputs;
    NDArray[] activationInputs;

    public NeuralNetwork(int numLayers, ActivationType[] activations, NDArray[] weights, int numInputs, int numOutputs, double[] log_stdVals,
        double alpha, double beta1, double beta2, double epsilon) {
        this.numLayers = numLayers;
        this.activations = activations;
        this.weights = weights;
        this.V_dw = new NDArray[weights.Length];
        this.S_dw = new NDArray[weights.Length];
        for (int i = 0; i < weights.Length; i++) {
            V_dw[i] = NDArray.NDArrayZeros(weights[i].shape);
            S_dw[i] = NDArray.NDArrayZeros(weights[i].shape);
        }
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        inputs = new NDArray[numLayers];
        activationInputs = new NDArray[numLayers];
        log_std = new NativeArray<double>(log_stdVals.Length, Allocator.Persistent);
        log_std_mean = 0;
        for (int i = 0; i < log_stdVals.Length; i++) {
            log_std[i] = log_stdVals[i];
            log_std_mean += log_stdVals[i];
        }
        log_std_mean /= log_stdVals.Length;
        iteration = 0;
    }

    public void resetOptimizerWeights() {
        for (int i = 0; i < weights.Length; i++) {
            V_dw[i] = NDArray.NDArrayZeros(weights[i].shape);
            S_dw[i] = NDArray.NDArrayZeros(weights[i].shape);
        }
    }

    public NDArray Forward(NDArray input) {
        inputs[0] = input;
        for (int i = 0; i < numLayers; i++) {
            //Step Forward Job
            NativeArray<double> nativeWeights = weights[i].getNativeArray(Allocator.TempJob);
            NativeArray<int> nativeWeightsShape = weights[i].getNativeShape(Allocator.TempJob);
            NativeArray<double> nativeInputs = inputs[i].getNativeArray(Allocator.TempJob);
            NativeArray<double> layerOutput = new NativeArray<double>(weights[i].shape[0], Allocator.TempJob);
            NativeArray<double> activationOutput = new NativeArray<double>(weights[i].shape[0], Allocator.TempJob);
            NNStepForwardJob stepForwardJob = new NNStepForwardJob {
                weights = nativeWeights,
                weightsShape = nativeWeightsShape,
                input = nativeInputs,
                activation = activations[i],
                layerOutput = layerOutput,
                activationOutput = activationOutput
            };
            JobHandle jobHandle = stepForwardJob.Schedule();
            jobHandle.Complete();
            
            //Save outputs of layer and activation function
            activationInputs[i] = NDArray.fromNativeArray(layerOutput);
            NDArray curOutput = NDArray.fromNativeArray(activationOutput);

            //Dispose all native arrays
            nativeWeights.Dispose();
            nativeWeightsShape.Dispose();
            nativeInputs.Dispose();
            layerOutput.Dispose();
            activationOutput.Dispose();

            if (i+1 == numLayers) {
                return curOutput;
            }
            inputs[i+1] = curOutput;
        }
        UnityEngine.Debug.Log("HUGE ERROR IN FORWARD FUNCTION. RETURNING INPUT AS OUTPUT!!!");
        return input;
    }

    public void Backward(NDArray grad) {
        NDArray gradient = grad;
        for (int i = numLayers - 1; i >= 0; i--) {  
            //Step Backward Job
            NativeArray<double> nativeWeights = weights[i].getNativeArray(Allocator.TempJob);
            NativeArray<int> nativeWeightsShape = weights[i].getNativeShape(Allocator.TempJob);
            NativeArray<double> nativeLayerInput = inputs[i].getNativeArray(Allocator.TempJob);
            NativeArray<double> nativeActivationInput = activationInputs[i].getNativeArray(Allocator.TempJob);
            NativeArray<double> nativeGrad = gradient.getNativeArray(Allocator.TempJob);
            NativeArray<double> nativeWeightsGradient = new NativeArray<double>(weights[i].numElements, Allocator.TempJob);
            NativeArray<double> nativeLayerGradient = new NativeArray<double>(weights[i].shape[1], Allocator.TempJob);
            NNStepBackwardJob stepBackwardJob = new NNStepBackwardJob {
                weights = nativeWeights,
                weightsShape = nativeWeightsShape,
                layerInput = nativeLayerInput,
                activationInput = nativeActivationInput,
                grad = nativeGrad,
                activation = activations[i],
                weightsGrad = nativeWeightsGradient,
                layerGrad = nativeLayerGradient
            };
            JobHandle jobHandle = stepBackwardJob.Schedule();
            jobHandle.Complete();

            gradient = NDArray.fromNativeArray(nativeLayerGradient);

            //Dispose redundant native arrays
            nativeWeightsShape.Dispose();
            nativeLayerInput.Dispose();
            nativeActivationInput.Dispose();
            nativeGrad.Dispose();
            nativeLayerGradient.Dispose();
            
            //Adam Optimization Job
            NativeArray<double> nativeV_dw = V_dw[i].getNativeArray(Allocator.TempJob);
            NativeArray<double> nativeS_dw = S_dw[i].getNativeArray(Allocator.TempJob);
            AdamOptimizerStepJob adamJob = new AdamOptimizerStepJob {
                weightsGrad = nativeWeightsGradient,
                alpha = alpha,
                beta1 = beta1,
                beta2 = beta2,
                epsilon = epsilon,
                iteration = iteration,
                V_dw = nativeV_dw,
                S_dw = nativeS_dw,
                weights = nativeWeights
            };
            jobHandle = adamJob.Schedule(weights.Length, 32);
            jobHandle.Complete();

            V_dw[i] = NDArray.fromNativeArray(nativeV_dw);
            S_dw[i] = NDArray.fromNativeArray(nativeS_dw);
            weights[i] = NDArray.fromNativeArray(nativeWeights);
            
            //Dispose all remaining native arrays
            nativeWeights.Dispose();
            nativeWeightsGradient.Dispose();
            nativeV_dw.Dispose();
            nativeS_dw.Dispose();
        }
        iteration++;
    }
}
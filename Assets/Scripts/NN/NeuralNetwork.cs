using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using System.Collections.Generic;
public struct NeuralNetwork { //Change to call small jobs to do all calcs
    
    //Adam Optimization Parameters
    private NativeArray<double>[] V_dw;
    private NativeArray<double>[] S_dw;
    private double alpha;
    private double beta1;
    private double beta2;
    private double epsilon;
    private int iteration;
    private int adamJobBatchSize;

    //Backward Gradients
    NativeArray<double>[] weightsGrads;
    NativeArray<double>[] layerGrads;

    //NN Parameters
    public int numLayers;
    public int numInputs;
    public int numOutputs;
    int maxForwardCalls;
    int maxBatchSize;

    ActivationType[] activations;
    public NativeArray<double> std;
    public double entropy;
    NativeArray<double>[,] inputs;
    public NativeArray<double>[,] activationInputs;
    public NativeArray<double>[] weights;
    public NativeArray<int>[] weightsShape;

    public NeuralNetwork(int numLayers, ActivationType[] activations, NativeArray<double>[] weights, NativeArray<int>[] weightsShape, int numInputs, int numOutputs, 
        double[] stdVals, double alpha, double beta1, double beta2, double epsilon, int adamJobBatchSize, int maxForwardCalls, int maxBatchSize) {
        this.numLayers = numLayers;
        this.maxForwardCalls = maxForwardCalls;
        this.maxBatchSize = maxBatchSize;
        this.activations = activations;
        this.weights = weights;
        this.weightsShape = weightsShape;
        weightsGrads = new NativeArray<double>[numLayers];
        layerGrads = new NativeArray<double>[numLayers];
        this.adamJobBatchSize = adamJobBatchSize;
        this.V_dw = new NativeArray<double>[numLayers];
        this.S_dw = new NativeArray<double>[numLayers];
        inputs = new NativeArray<double>[maxForwardCalls, numLayers];
        activationInputs = new NativeArray<double>[maxForwardCalls, numLayers];
        for (int i = 0; i < maxForwardCalls; i++) {
            for (int j = 0; j < numLayers; j++) {
                inputs[i, j] = new NativeArray<double>(weightsShape[j][1]*maxBatchSize, Allocator.Persistent);
                activationInputs[i, j] = new NativeArray<double>(weightsShape[j][0]*maxBatchSize, Allocator.Persistent);
            }
        }
        
        for (int i = 0; i < numLayers; i++) {
            V_dw[i] = new NativeArray<double>(weights[i].Length, Allocator.Persistent);
            S_dw[i] = new NativeArray<double>(weights[i].Length, Allocator.Persistent);
            for (int j = 0; j < weights[i].Length; j++) {
                V_dw[i][j] = 0;
                S_dw[i][j] = 0;
            }
        }
        iteration = 1;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        
        std = new NativeArray<double>(stdVals.Length, Allocator.Persistent);
        double std_mean = 0;
        for (int i = 0; i < stdVals.Length; i++) {
            std[i] = stdVals[i];
            std_mean += stdVals[i];
        }
        std_mean /= stdVals.Length;
        entropy = GaussianDistribution.entropy(std_mean);
    }

    public void resetOptimizerWeights() {
        for (int i = 0; i < numLayers; i++) {
            for (int j = 0; j < weights[i].Length; j++) {
                V_dw[i][j] = 0;
                S_dw[i][j] = 0;
            }
        }
        iteration = 1;
    }

    public JobHandle Forward(NativeArray<double> input, int numInputs, NativeArray<double> output, int id) {
        NativeNDOps.appendOnes(inputs[id, 0], input, numInputs, weightsShape[0][1]-1);
        JobHandle prevLayerHandle = new JobHandle();
        for (int i = 0; i < numLayers; i++) {
            //Step Forward Job
            NativeArray<double> activationOutput;
            int appendOnes = 1;
            if (i+1 == numLayers) {
                activationOutput = output;
                appendOnes = 0;
            } else {
                activationOutput = inputs[id, i+1];
            }
            NNStepForwardJob stepForwardJob = new NNStepForwardJob {
                weights = weights[i],
                weightsShape = weightsShape[i],
                input = inputs[id, i],
                numInputs = numInputs,
                activation = activations[i],
                layerOutput = activationInputs[id, i],
                activationOutput = activationOutput,
                shouldAppendOnes = appendOnes
            };

            JobHandle curLayerHandle;
            if (i == 0) {
                curLayerHandle = stepForwardJob.Schedule();
            } else {
                curLayerHandle = stepForwardJob.Schedule(prevLayerHandle);
            }
            if (i+1 == numLayers) {
                return curLayerHandle;
            }
            prevLayerHandle = curLayerHandle;
        }
        UnityEngine.Debug.Log("HUGE ERROR IN FORWARD FUNCTION. RETURNING INPUT AS OUTPUT!!!");
        return new JobHandle();
    }

    public JobHandle Forward(NativeArray<double> input, int numInputs, NativeArray<double> output, int id, JobHandle dependency) {
        NativeNDOps.appendOnes(inputs[id, 0], input, numInputs, weightsShape[0][1]-1);
        JobHandle prevLayerHandle = new JobHandle();
        for (int i = 0; i < numLayers; i++) {
            //Step Forward Job
            NativeArray<double> activationOutput;
            int appendOnes = 1;
            if (i+1 == numLayers) {
                activationOutput = output;
                appendOnes = 0;
            } else {
                activationOutput = inputs[id, i+1];
            }
            NNStepForwardJob stepForwardJob = new NNStepForwardJob {
                weights = weights[i],
                weightsShape = weightsShape[i],
                input = inputs[id, i],
                numInputs = numInputs,
                activation = activations[i],
                layerOutput = activationInputs[id, i],
                activationOutput = activationOutput,
                shouldAppendOnes = appendOnes
            };

            JobHandle curLayerHandle;
            if (i == 0) {
                curLayerHandle = stepForwardJob.Schedule(dependency);
            } else {
                curLayerHandle = stepForwardJob.Schedule(prevLayerHandle);
            }
            if (i+1 == numLayers) {
                return curLayerHandle;
            }
            prevLayerHandle = curLayerHandle;
        }
        UnityEngine.Debug.Log("HUGE ERROR IN FORWARD FUNCTION. RETURNING INPUT AS OUTPUT!!!");
        return new JobHandle();
    }

    public JobHandle Backward(NativeArray<double> grad, int numGrads, int id) {
        JobHandle prevLayerHandle = new JobHandle();
        JobHandle curAdamHandle = new JobHandle();
        NativeArray<double> gradient;
        for (int i = numLayers - 1; i >= 0; i--) {
            //Step Backward Job
            if (i == numLayers - 1) {
                gradient = grad;
            } else {
                gradient = layerGrads[i+1];
            }
            weightsGrads[i] = new NativeArray<double>(weights[i].Length, Allocator.TempJob);
            layerGrads[i] = new NativeArray<double>((weightsShape[i][1]-1)*numGrads, Allocator.TempJob);
            NNStepBackwardJob stepBackwardJob = new NNStepBackwardJob {
                weights = weights[i],
                weightsShape = weightsShape[i],
                layerInput = inputs[id, i],
                activationInput = activationInputs[id, i],
                grad = gradient,
                numGrads = numGrads,
                activation = activations[i],
                weightsGrad = weightsGrads[i],
                layerGrad = layerGrads[i]
            };

            JobHandle curLayerHandle;
            if (i == numLayers - 1) {
                curLayerHandle = stepBackwardJob.Schedule();
            } else {
                curLayerHandle = stepBackwardJob.Schedule(prevLayerHandle);
            }
            
            //Adam Optimization Job
            AdamOptimizerStepJob adamJob = new AdamOptimizerStepJob {
                weightsGrad = weightsGrads[i],
                alpha = (alpha/math.sqrt(GameManager.ITERATION)),
                beta1 = beta1,
                beta2 = beta2,
                epsilon = epsilon,
                iteration = iteration,
                V_dw = V_dw[i],
                S_dw = S_dw[i],
                weights = weights[i]
            };

            curAdamHandle = adamJob.Schedule(weights[i].Length, adamJobBatchSize, curLayerHandle);
            prevLayerHandle = curAdamHandle;
        }
        iteration++;
        return curAdamHandle;
    }
    
    public JobHandle Backward(NativeArray<double> grad, int numGrads, int id, JobHandle dependency) {
        JobHandle prevLayerHandle = new JobHandle();
        JobHandle curAdamHandle = new JobHandle();
        NativeArray<double> gradient;
        for (int i = numLayers - 1; i >= 0; i--) {
            //Step Backward Job
            if (i == numLayers - 1) {
                gradient = grad;
            } else {
                gradient = layerGrads[i+1];
            }
            weightsGrads[i] = new NativeArray<double>(weights[i].Length, Allocator.TempJob);
            layerGrads[i] = new NativeArray<double>((weightsShape[i][1]-1)*numGrads, Allocator.TempJob);
            NNStepBackwardJob stepBackwardJob = new NNStepBackwardJob {
                weights = weights[i],
                weightsShape = weightsShape[i],
                layerInput = inputs[id, i],
                activationInput = activationInputs[id, i],
                grad = gradient,
                numGrads = numGrads,
                activation = activations[i],
                weightsGrad = weightsGrads[i],
                layerGrad = layerGrads[i]
            };

            JobHandle curLayerHandle;
            if (i == numLayers - 1) {
                curLayerHandle = stepBackwardJob.Schedule(dependency);
            } else {
                curLayerHandle = stepBackwardJob.Schedule(prevLayerHandle);
            }
            
            //Adam Optimization Job
            AdamOptimizerStepJob adamJob = new AdamOptimizerStepJob {
                weightsGrad = weightsGrads[i],
                alpha = (alpha/math.sqrt(GameManager.ITERATION)),
                beta1 = beta1,
                beta2 = beta2,
                epsilon = epsilon,
                iteration = iteration,
                V_dw = V_dw[i],
                S_dw = S_dw[i],
                weights = weights[i]
            };
            curAdamHandle = adamJob.Schedule(weights[i].Length, adamJobBatchSize, curLayerHandle);
            prevLayerHandle = curAdamHandle;
        }
        iteration++;
        return curAdamHandle;
    }

    public void Dispose() {
        for (int i = 0; i < numLayers; i++) {
            weightsShape[i].Dispose();
            weights[i].Dispose();
            for (int j = 0; j < maxForwardCalls; j++) {
                if (i != 0) {
                    inputs[j, i].Dispose();
                }
                activationInputs[j, i].Dispose();
            }
            V_dw[i].Dispose();
            S_dw[i].Dispose();
        }
        std.Dispose();
    }

    public void resetGrads() {
        for (int i = 0; i < numLayers; i++) {
            weightsGrads[i].Dispose();
            layerGrads[i].Dispose();
        }
    }
}
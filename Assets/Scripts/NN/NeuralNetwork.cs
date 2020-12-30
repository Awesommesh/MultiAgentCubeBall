using Unity.Collections;
using Unity.Mathematics;
public struct NeuralNetwork {
    NativeArray<NDArray> weights;
    //Adam Optimization Parameters
    private NativeArray<NDArray> V_dw;
    private NativeArray<NDArray> S_dw;
    private double alpha;
    private double beta1;
    private double beta2;
    private double epsilon;
    private int iteration;
    int numLayers;
    NativeArray<ActivationType> activations;
    int numInputs;
    int numOutputs;
    public NDArray log_std;
    NativeArray<NDArray> inputs;
    NativeArray<NDArray> activationInputs;

    public NeuralNetwork(int numLayers, NativeArray<ActivationType> activations, NativeArray<NDArray> weights, int numInputs, int numOutputs, NativeArray<double> log_stdVals,
        double alpha, double beta1, double beta2, double epsilon) {
        this.numLayers = numLayers;
        this.activations = activations;
        this.weights = weights;
        this.V_dw = new NativeArray<NDArray>(weights.Length, Allocator.Persistent);
        this.S_dw = new NativeArray<NDArray>(weights.Length, Allocator.Persistent);
        for (int i = 0; i < weights.Length; i++) {
            V_dw[i] = NDArray.NDArrayZeros(weights[i].shape, Allocator.Persistent);
            S_dw[i] = NDArray.NDArrayZeros(weights[i].shape, Allocator.Persistent);
        }
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        inputs = new NativeArray<NDArray>(numLayers, Allocator.Persistent);
        activationInputs = new NativeArray<NDArray>(numLayers, Allocator.Persistent);
        NativeArray<int> log_stdShape = new NativeArray<int>(1, Allocator.Persistent);
        log_stdShape[0] = numOutputs;
        log_std = new NDArray(log_stdShape, log_stdVals, Allocator.Persistent);
        iteration = 0;
    }

    public void resetOptimizerWeights() {
        for (int i = 0; i < weights.Length; i++) {
            V_dw[i] = NDArray.NDArrayZeros(weights[i].shape, Allocator.Persistent);
            S_dw[i] = NDArray.NDArrayZeros(weights[i].shape, Allocator.Persistent);
        }
    }

    public void Dispose() {
        for (int i = 0; i < weights.Length; i++) {
            weights[i].Dispose();
            V_dw[i].Dispose();
            S_dw[i].Dispose();
        }
        weights.Dispose();
        V_dw.Dispose();
        S_dw.Dispose();
        for (int i = 0; i < inputs.Length; i++) {
            inputs[i].Dispose();
            activationInputs[i].Dispose();
        }
        activations.Dispose();
        activationInputs.Dispose();
        inputs.Dispose();
    }

    public NDArray Forward(NDArray input) {
        inputs[0] = input;
        for (int i = 0; i < numLayers; i++) {
            activationInputs[i] = NDArray.Dot(weights[i], inputs[i]);
            NDArray curOutput;
            switch(activations[i]) {
                case ActivationType.ReLU:
                    curOutput = activationInputs[i] * (activationInputs[i] > 0);
                    break;
                case ActivationType.Sigmoid:
                    curOutput = Sigmoid(activationInputs[i]);
                    break;
                default:
                    curOutput = activationInputs[i];
                    break;
            }
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
            NDArray curGrad;
            switch(activations[i]) {
                case ActivationType.ReLU:
                    curGrad = (activationInputs[i] > 0) * gradient;
                    break;
                case ActivationType.Sigmoid:
                    curGrad = Sigmoid(activationInputs[i]) * gradient;
                    break;
                default:
                    curGrad = gradient;
                    break;
            }
            NDArray weightsGrad = NDArray.Dot(curGrad, inputs[i].T());
            gradient = weights[i].T() * curGrad;

            //Update weights[i] using Adam Optimization
            V_dw[i] = (beta1 * V_dw[i]) + (1-beta1) * weightsGrad;
            S_dw[i] = (beta2 * S_dw[i]) + (1-beta2) * NDArray.Pow(weightsGrad, 2);
            NDArray V_dw_corrected = V_dw[i] / (1 - math.pow(beta1, iteration));
            NDArray S_dw_corrected = S_dw[i] / (1 - math.pow(beta2, iteration));
            weights[i] -= alpha * (V_dw_corrected / (NDArray.Sqrt(S_dw_corrected) + epsilon));
        }
        iteration++;
    }

    private NDArray Sigmoid(NDArray x) {
        return NDArray.Exp(x) / (NDArray.Exp(x) + 1);
    }
}
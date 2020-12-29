using Unity.Collections;
public struct NeuralNetwork {
    int numLayers;
    NativeArray<ActivationType> activations;
    NativeArray<NDArray> weights;
    int numInputs;
    int numOutputs;
    NativeArray<NDArray> inputs;
    NativeArray<NDArray> activationInputs;

    public NeuralNetwork(int numLayers, NativeArray<ActivationType> activations, NativeArray<NDArray> weights, int numInputs, int numOutputs) {
        this.numLayers = numLayers;
        this.activations = activations;
        this.weights = weights;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        inputs = new NativeArray<NDArray>(numLayers, Allocator.Persistent);
        activationInputs = new NativeArray<NDArray>(numLayers, Allocator.Persistent);
    }

    public void Dispose() {
        for (int i = 0; i < weights.Length; i++) {
            weights[i].Dispose();
        }
        for (int i = 0; i < inputs.Length; i++) {
            inputs[i].Dispose();
            activationInputs[i].Dispose();
        }
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
                    curOutput = NDArray.Exp(activationInputs[i]) / (NDArray.Exp(activationInputs[i]) + 1);
                    break;
                default:
                    curOutput = activationInputs[i] * (activationInputs[i] > 0);
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
}
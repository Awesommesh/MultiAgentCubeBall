using Unity.Entities;
public struct FullyConnected : IComponentData {
    public NNLayer type {
        get {
            return NNLayer.FullyConnected;
        }
    }
    int InputDim {
        get;
        set;
    }

    int OutputDim {
        get;
        set;
    }
    TwoDArray Input {
        get;
        set;
    }

    TwoDArray Output {
        get;
        set;
    }

    TwoDArray Parameters {
        get;
        set;
    }
    ActivationFunction Activation {
        get;
        set;
    }
    public FullyConnected(int numInput, int numOutput, ActivationFunctionType act, uint seed) {
        Parameters = Operations.Random2DArray(numInput, numOutput, seed);
        this.Input = new TwoDArray(numInput, 1);
        this.Output = new TwoDArray(numOutput, 1);
        this.InputDim = numInput;
        this.OutputDim = numOutput;
        Activation = new ActivationFunction(act, numOutput);
    }

    public void Forward(TwoDArray Input) {
        this.Input = Input;
        this.Output = Activation.Forward(Operations.Dot(Input, Parameters));
    }
}

using Unity.Entities;
public struct FullyConnected : IComponentData {
    public string name {
        get {
            return "Fully Connected";
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

    IActivation Activation {
        get;
        set;
    }
    public FullyConnected(int numInput, int numOutput, string act, uint seed) {
        Parameters = Operations.Random2DArray(numInput, numOutput, seed);
        this.Input = new TwoDArray(numInput, 1);
        this.Output = new TwoDArray(numOutput, 1);
        this.InputDim = numInput;
        this.OutputDim = numOutput; 
        Activation = BaseActivation.getActivation(act, numOutput);
    }

    public void Forward(TwoDArray Input) {
        this.Input = Input;
        this.Output = Activation.Forward(Operations.Dot(Input, Parameters));
    }
}

using Unity.Entities;

public struct ActivationFunction : IComponentData {
    public ActivationFunctionType type {
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

    public ActivationFunction(ActivationFunctionType type, int numInput) {
        this.type = type;
        this.Input = new TwoDArray(numInput, 1);
        this.Output = new TwoDArray(numInput, 1);
    }

    public TwoDArray Forward(TwoDArray Input) {
        this.Input = Input;
        switch(type) {
            case ActivationFunctionType.ReLU:
                Output = Input * (Input > 0);
                break;
            case ActivationFunctionType.Sigmoid:
                Output = Operations.Exp(Input) / (1 + Operations.Exp(Input));
                break;
            default:
                Output = Input;
                break;
        }
        
        return Output;
    }
}
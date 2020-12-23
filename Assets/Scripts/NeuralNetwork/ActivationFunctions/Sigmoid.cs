using Unity.Entities;

public struct Sigmoid : IComponentData, IActivation {
    public string Name {
        get {
            return "Sigmoid";
        }
    }

    TwoDArray Input {
        get;
        set;
    }

    TwoDArray Output {
        get;
        set;
    }

    public Sigmoid(int numInput) {
        this.Input = new TwoDArray(numInput, 1);
        this.Output = new TwoDArray(numInput, 1);
    }

    public TwoDArray Forward(TwoDArray Input) {
        this.Input = Input;
        Output = Operations.Exp(Input) / (1 + Operations.Exp(Input));
        return Output;
    }
}
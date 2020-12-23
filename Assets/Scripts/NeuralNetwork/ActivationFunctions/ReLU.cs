using Unity.Entities;

public struct ReLU : IComponentData, IActivation {
    public string Name {
        get {
            return "ReLU";
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

    public ReLU(int numInput) {
        this.Input = new TwoDArray(numInput, 1);
        this.Output = new TwoDArray(numInput, 1);
    }

    public TwoDArray Forward(TwoDArray Input) {
        this.Input = Input;
        Output = Input * (Input > 0);
        return Output;
    }
}
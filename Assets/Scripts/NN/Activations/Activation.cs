/*using Unity.Collections;
public struct Activation {
    public int numInput {
        get;
        set;
    }
    public NDArray input {
        get;
        set;
    }

    public NDArray output {
        get;
        set;
    }

    public ActivationType type {
        get;
        set;
    }

    public Activation(ActivationType type, int numInput, Allocator allocator) { 
        this.type = type;
        this.numInput = numInput;
        this.input = null;
    }

    public void Forward(NDArray input) {
        this.input = input;
        switch(type) {
            case ActivationType.ReLU:
                output = input * (input > 0);
                break;
            case ActivationType.Sigmoid:
                output = NDArray.Exp(input) / (NDArray.Exp(input) + 1);
                break;
            default: //RELU for now
                output = input * (input > 0);
                break;
        }
    }

    public void Dispose() {
        input.Dispose();
        output.Dispose();
    }
}*/
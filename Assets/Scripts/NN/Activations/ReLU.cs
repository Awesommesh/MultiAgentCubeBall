public struct ReLU {
    public NDArray input {
        get;
        set;
    }

    public NDArray output {
        get;
        set;
    }

    public void Forward (NDArray input) {
        this.input = input;
        output = input * (input > 0);
    }
}
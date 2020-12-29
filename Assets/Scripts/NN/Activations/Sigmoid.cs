public struct Sigmoid {
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
        output = NDArray.Exp(input) / (NDArray.Exp(input) + 1);
    }
}
using Unity.Entities;

public struct BaseActivation : IComponentData {
    public static IActivation getActivation(string name, int numInput) {
        switch(name) {
            case "relu":
                return new ReLU(numInput);
            case "sigmoid":
                return new Sigmoid(numInput);
            default://Temporarily ReLU is defautl
                return new ReLU(numInput);
        }
    }
}

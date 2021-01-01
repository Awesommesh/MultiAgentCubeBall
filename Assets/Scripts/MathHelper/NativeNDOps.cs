using Unity.Entities;
using Unity.Mathematics;
using Unity.Collections;
public struct NativeNDOps : IComponentData {
    //No Check for incorrect dimensions!!!
    public static void Dot(NativeArray<double> a, NativeArray<int> aShape, NativeArray<double> b, NativeArray<int> bShape, NativeArray<double> output) {
        //output shape = [aShape[0], bShape[1]];
        for (int i = 0; i < aShape[0]; i++) {
            for (int j = 0; j < bShape[1]; j++) {
                int ind = i*bShape[1] + j;
                output[ind] = 0;
                for (int k = 0; k < aShape[1]; k++) {
                    output[ind] += a[i*aShape[1]+k] * b[k*bShape[1] + j];
                }
            }
        }
    }

    //Only works for 2D matrix
    public static void Transpose(NativeArray<double> a, NativeArray<int> aShape, NativeArray<double> output) {
        for (int i = 0; i < aShape[0]; i++) {
            for (int j = 0; j < aShape[1]; j++) {
                output[j*aShape[0]+i] = a[i*aShape[1] + j];
            }
        }
    }

    public static void ActivationFunction(ActivationType type, NativeArray<double> input, NativeArray<double> output) {
        switch(type) {
            case ActivationType.ReLU:
                for (int i = 0; i < output.Length; i++) {
                    output[i] = ReLU(input[i]);
                }
                break;
            case ActivationType.Sigmoid:
                for (int i = 0; i < output.Length; i++) {
                    output[i] = Sigmoid(input[i]);
                }
                break;
            default:
                for (int i = 0; i < output.Length; i++) {
                    output[i] = input[i];
                }
                break;
        }
    }

    public static void ActivationFunctionBack(ActivationType type, NativeArray<double> input, NativeArray<double> grad, NativeArray<double> output) {
        switch(type) {
            case ActivationType.ReLU:
                for (int i = 0; i < output.Length; i++) {
                    output[i] = ReLU_back(input[i])*grad[i];
                }
                break;
            case ActivationType.Sigmoid:
                for (int i = 0; i < output.Length; i++) {
                    output[i] = Sigmoid_back(input[i])*grad[i];
                }
                break;
            default:
                for (int i = 0; i < output.Length; i++) {
                    output[i] = grad[i];
                }
                break;
        }
    }

    public static double ReLU(double x) {
        if (x > 0) {
            return x;
        }
        return 0;
    }

    public static double Sigmoid(double x) {
        return 1 / (math.exp(-x) + 1);
    }

    public static double ReLU_back(double x) {
        if (x > 0) {
            return 1;
        }
        return 0;
    }

    public static double Sigmoid_back(double x) {
        return Sigmoid(x) * (1 - Sigmoid(x));
    }

    public static void Copy(NativeArray<double> input, NativeArray<double> output) {
        for (int i = 0; i < input.Length; i++) {
            output[i] = input[i];
        }
    }
}
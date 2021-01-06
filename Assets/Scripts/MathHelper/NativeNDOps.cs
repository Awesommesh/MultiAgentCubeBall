using Unity.Entities;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;

public struct NativeNDOps : IComponentData {
    [BurstCompile]
    public static NativeArray<double> HeInitializedNDArray(NativeArray<int> shape, int prevSize, Allocator allocator) {
        int numElements = 1;
        for (int i = 0; i < shape.Length; i++) {
            numElements *= shape[i];
        }
        NativeArray<double> he = new NativeArray<double>(numElements, allocator);
        for (int i = 0; i < numElements; i++) {
            he[i] = GaussianDistribution.NextGaussian() * math.sqrt(((double)2)/prevSize);
        }
        return he;
    }

    //No Check for incorrect dimensions!!!
    //[BurstCompile]
    public static unsafe void Dot(NativeArray<double> a, NativeArray<int> aShape, int aTranspose, NativeArray<double> b, NativeArray<int> bShape, int bTranspose, NativeArray<double> output) {
        matmul(aTranspose, bTranspose, aShape[0], aShape[1], bShape[0], bShape[1], (double*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(a), (double*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(b), (double*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(output));
        /*//output shape = [aShape[0], bShape[1]];
        int aShape0 = aShape[0];
        int aShape1 = aShape[1];
        int bShape0 = bShape[0];
        int bShape1 = bShape[1];
        NativeArray<double> aDot = a;
        NativeArray<double> bDot = b;
        if (aTranspose == 1 && bTranspose == 1) { //Transpose A
            aDot = new NativeArray<double>(a.Length, Allocator.Temp);
            for (int i = 0; i < aShape0; i++) {
                int aInd = i * aShape1;
                for (int j = 0; j < aShape1; j++) {
                    aDot[j*aShape0+i] = a[aInd + j];
                }
            }
            int temp = aShape0;
            aShape0 = aShape1;
            aShape1 = temp;
        } else if (aTranspose == 1 && bTranspose != 1) { //Transpose Both
            aDot = new NativeArray<double>(a.Length, Allocator.Temp);
            for (int i = 0; i < aShape0; i++) {
                int aInd = i * aShape1;
                for (int j = 0; j < aShape1; j++) {
                    aDot[j*aShape0+i] = a[aInd + j];
                }
            }
            bDot = new NativeArray<double>(b.Length, Allocator.Temp);
            for (int i = 0; i < bShape0; i++) {
                int bInd = i * bShape1;
                for (int j = 0; j < bShape1; j++) {
                    bDot[j*bShape0+i] = b[bInd + j];
                }
            }
            int temp = aShape0;
            aShape0 = aShape1;
            aShape1 = temp;
            temp = bShape0;
            bShape0 = bShape1;
            bShape1 = temp;
        } else if (aTranspose != 1 && bTranspose == 1) { //No Transpose needed
            //Do Nothing
        } else { // Transpose B
            bDot = new NativeArray<double>(b.Length, Allocator.Temp);
            for (int i = 0; i < bShape0; i++) {
                int bInd = i * bShape1;
                for (int j = 0; j < bShape1; j++) {
                    bDot[j*bShape0+i] = b[bInd + j];
                }
            }
            int temp = bShape0;
            bShape0 = bShape1;
            bShape1 = temp;
        }
        for (int i = 0; i < aShape0; i++) {
            for (int j = 0; j < bShape0; j++) {
                int ind = i*bShape0 + j;
                output[ind] = 0;
                int aInd = i*aShape1;
                int bInd = j*aShape1;
                for (int k = 0; k < aShape1; k++) {
                    output[ind] += a[aInd+k] * b[bInd + k];
                }
            }
        }
        if (aTranspose == 1) {
            aDot.Dispose();
        }
        if (bTranspose != 1) {
            bDot.Dispose();
        }*/
    }

    [DllImport("/Users/animeshagrawal/repositories/MultiAgentCubeBall/Assets/Scripts/MathHelper/matmul.dylib")]
    public static extern unsafe void matmul(int transA, int transB, int A_width, int A_height, int B_width, int B_height, 
        double* A, double* B, double* output);

    //Only works for 2D matrix
    [BurstCompile]
    public static void Transpose(NativeArray<double> a, NativeArray<int> aShape, NativeArray<double> output) {
        for (int i = 0; i < aShape[0]; i++) {
            for (int j = 0; j < aShape[1]; j++) {
                output[j*aShape[0]+i] = a[i*aShape[1] + j];
            }
        }
    }

    [BurstCompile]
    public static double ActivationFunction(ActivationType type, double input) {
        switch(type) {
            case ActivationType.ReLU:
                return ReLU(input);
            case ActivationType.Sigmoid:
                return Sigmoid(input);
            default:
                return input;
        }
    }

    [BurstCompile]
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

    [BurstCompile]
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

    [BurstCompile]
    public static double ReLU(double x) {
        if (x > 0) {
            return x;
        }
        return 0;
    }

    [BurstCompile]
    public static double Sigmoid(double x) {
        return 1 / (math.exp(-x) + 1);
    }

    [BurstCompile]
    public static double ReLU_back(double x) {
        if (x > 0) {
            return 1;
        }
        return 0;
    }

    [BurstCompile]
    public static double Sigmoid_back(double x) {
        return Sigmoid(x) * (1 - Sigmoid(x));
    }

    [BurstCompile]
    public static void Copy(NativeArray<double> input, NativeArray<double> output) {
        for (int i = 0; i < input.Length; i++) {
            output[i] = input[i];
        }
    }

    [BurstCompile]
    public unsafe static void appendOnes(NativeArray<double> inputArray, NativeArray<double> inputData, int numInputs, int inputSize) {
        UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafePtr(inputArray),
				NativeArrayUnsafeUtility.GetUnsafePtr(inputData), 
                numInputs * inputSize * (long) UnsafeUtility.SizeOf<double>());
        int lastRow = inputSize*numInputs;
        for (int i = 0; i  < numInputs; i++) {
            inputArray[lastRow + i] = 1;
        }
    }

    [BurstCompile]
    public unsafe static void CopyPartial(NativeArray<double> input, NativeArray<double> output, int length) {
        UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafePtr(output),
				NativeArrayUnsafeUtility.GetUnsafePtr(input), 
                length * (long) UnsafeUtility.SizeOf<double>());
    }

    //For debugging NativeArrays
    public static void Print(NativeArray<double> data, NativeArray<int> shape){
        string str = "";
        for (var i = 0; i<shape[0]; i++){
            for (var j=0; j<shape[1]; j++){
                str += data[i*shape[1]+j] + "\t";
            }
            str += "\n";
        }
        UnityEngine.Debug.Log(str);
    }
}
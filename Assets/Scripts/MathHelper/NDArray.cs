using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
public struct NDArray {
    private const double EPSILON = 0.00000000001;
    private static Unity.Mathematics.Random rng = new Unity.Mathematics.Random(GameManager.NDArrayGenSeed);
    private double[] array;

    public int[] shape {
        get;
        set;
    }

    public int numElements {
        get;
        set;
    }

    public NativeArray<int> getNativeShape(Allocator allocator) {
        NativeArray<int> nativeShape = new NativeArray<int>(shape.Length, allocator);
        for (int i = 0; i < shape.Length; i++) {
            nativeShape[i] = shape[i];
        }
        return nativeShape;
    }

    public NativeArray<double> getNativeArray(Allocator allocator) {
        NativeArray<double> nativeArray = new NativeArray<double>(numElements, allocator);
        for (int i = 0; i < numElements; i++) {
            nativeArray[i] = array[i];
        }
        return nativeArray;
    }

    public static NDArray fromNativeArray(NativeArray<double> data) {
        NDArray fromNative = new NDArray(data.Length, 1);
        for (int i = 0; i < data.Length; i++) {
            fromNative[i] = data[i];
        } 
        return fromNative;
    }

    public static NDArray fromNativeArray(NativeArray<double> data, int index, int len) {
        NDArray fromNative = new NDArray(len, 1);
        for (int i = index; i < index+len; i++) {
            fromNative[i] = data[i];
        } 
        return fromNative;
    }

    public void fillNativeArray(NativeArray<double> data, int index, int len) {
        for (int i = index; i < index + len; i++) {
            data[i] = this[i-index];
        }
    }

    public static NDArray NDArrayZeros(params int[] shape) {
        NDArray zeros = new NDArray(shape);
        for (int i = 0; i < zeros.numElements; i++) {
            zeros[i] = 0;
        }
        return zeros;
    }

    public static NDArray RandomNDArray(params int[] shape) {
        NDArray random = new NDArray(shape);
        for (int i = 0; i < random.numElements; i++) {
            random[i] = rng.NextDouble();
        }
        return random;
    }

    public NDArray (params int[] shape) {
        this.shape = shape;
        int e = 1;
        for (int i = 0; i < shape.Length; i++) {
            e *= shape[i];
        }
        this.numElements = e;
        this.array = new double[e];
    }

    public void Fill(double[] data) {
        for (int i = 0; i < data.Length; i++) {
            this.array[i] = data[i];
        }
    }

    public double this[params int[] indices] {
        get {
            if (indices.Length == 1) {
                return array[indices[0]];
            } else {
                int[] strides = getStrides();
                int index = 0;
                for (int i = 0; i < indices.Length; i++) {
                    index += indices[i] * strides[i];
                }
                return array[index];
            }
        }

        set {
            if (indices.Length == 1) {
                array[indices[0]] = value;
            } else {
                int[] strides = getStrides();
                int index = 0;
                for (int i = 0; i < indices.Length; i++) {
                    index += indices[i] * strides[i];
                }
                array[index] = value;
            }
        }
    }

    //Operations
    public static NDArray operator > (NDArray a, int b) {
        NDArray greaterThan = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            greaterThan[i] = a[i] > b ? 1 : 0;
        }
        return greaterThan;
    }

    public static NDArray operator < (NDArray a, int b) {
        NDArray lessThan = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            lessThan[i] = a[i] < b ? 1 : 0;
        }
        return lessThan;
    }

    public static NDArray operator > (NDArray a, NDArray b) {
        NDArray greaterThan = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            greaterThan[i] = a[i] > b[i] ? 1 : 0;
        }
        return greaterThan;
    }

    public static NDArray operator < (NDArray a, NDArray b) {
        NDArray lessThan = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            lessThan[i] = a[i] < b[i] ? 1 : 0;
        }
        return lessThan;
    }

    public static NDArray operator * (NDArray a, NDArray b) {
        NDArray product = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i]*b[i];
        }
        return product;
    }

    public static NDArray operator * (NDArray a, double b) {
        NDArray product = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i]*b;
        }
        return product;
    }

    public static NDArray operator * (double b, NDArray a) {
        NDArray product = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i]*b;
        }
        return product;
    }

    public static NDArray operator / (NDArray a, NDArray b) {
        NDArray dividend = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            if (b[i] == 0) {
                dividend[i] = a[i] / EPSILON;
            } else {
                dividend[i] = a[i]/b[i];
            }
        }
        return dividend;
    }

    public static NDArray operator / (NDArray a, double b) {
        NDArray dividend = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            if (b == 0) {
                dividend[i] = a[i] / EPSILON;
            } else {
                dividend[i] = a[i]/b;
            }
        }
        return dividend;
    }

    public static NDArray operator / (double b, NDArray a) {
        NDArray dividend = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            if (b == 0) {
                dividend[i] = a[i] / EPSILON;
            } else {
                dividend[i] = a[i]/b;
            }
        }
        return dividend;
    }

    public static NDArray operator + (NDArray a, double b) {
        NDArray sum = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }
    public static NDArray operator + (double b, NDArray a) {
        NDArray sum = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }

    public static NDArray operator + (NDArray a, NDArray b) {
        NDArray sum = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b[i];
        }
        return sum;
    }

    public static NDArray operator - (NDArray a, double b) {
        NDArray difference = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            difference[i] = a[i] - b;
        }
        return difference;
    }
    public static NDArray operator - (double b, NDArray a) {
        NDArray difference = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            difference[i] = b - a[i];
        }
        return difference;
    }

    public static NDArray operator - (NDArray a, NDArray b) {
        NDArray difference = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            difference[i] = a[i] - b[i];
        }
        return difference;
    }

    public static NDArray Exp (NDArray a) {
        NDArray exp = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            exp[i] = math.exp(a[i]);
        }
        return exp;
    }

    public static NDArray Sqrt (NDArray a) {
        NDArray sqrt = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            sqrt[i] = math.sqrt(a[i]);
        }
        return sqrt;
    }
    public static NDArray Dot (NDArray a, NDArray b) {
        NDArray dot = new NDArray(a.shape);
        for (int i = 0; i < a.shape[0]; i++) {
            for (int j = 0; j < b.shape[1]; j++) {
                dot[i, j] = 0;
                for (int k = 0; k < a.shape[1]; k++) {
                    dot[i, j] += a[i, k] * b[k, j];
                }
            }
        }
        return dot;
    }

    public static NDArray Clamp(NDArray a, double min, double max) {
        NDArray clamp = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            if (a[i] < min) {
                clamp[i] = min;
            } else if (a[i] > max) {
                clamp[i] = max;
            } else {
                clamp[i] = a[i];
            }
        }
        return clamp;
    }

    public static NDArray Clamp_Back(NDArray a, double min, double max) {
        NDArray clamp_back = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            if (a[i] < min) {
                clamp_back[i] = 0;
            } else if (a[i] > max) {
                clamp_back[i] = 0;
            } else {
                clamp_back[i] = 1;
            }
        }
        return clamp_back;
    }

    public static NDArray Min(NDArray a, NDArray b) {
        NDArray min = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            min[i] = math.min(a[i], b[i]);
        }
        return min;
    }

    public static NDArray Pow(NDArray a, double p) {
        NDArray pow = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            pow[i] = math.pow(a[i], p);
        }
        return pow;
    }
    public double Mean() {
        double mean = 0;
        for (int i = 0; i < numElements; i++) {
            mean += array[i];
        }
        return mean/numElements;
    }

    public NDArray T() {
        NDArray transpose = new NDArray(this.shape[1], this.shape[0]);
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                transpose[j, i] = this[i, j];
            }
        }
        return transpose;
    }

    public static NDArray Copy(NDArray a) {
        NDArray copy = new NDArray(a.shape);
        for (int i = 0; i < a.numElements; i++) {
            copy[i] = a[i];
        }
        return copy;
    }

    private int[] getStrides() {
        int mult = 1;
        int[] strides = new int[shape.Length];
        for (int i = shape.Length-1; i >= 0; i--) {
            strides[i] = mult;
            mult *= shape[i];
        }
        return strides;
    }

    //For debugging 2D NDArrays
    public void Print() {
        string toString = "";
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                toString += this[i, j] + "\t";
            }
            toString += "\n";
        }
        Debug.Log(toString);
    }
}
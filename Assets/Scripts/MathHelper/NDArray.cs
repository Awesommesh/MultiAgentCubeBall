using Unity.Collections;
using Unity.Mathematics;
public struct NDArray {
    private static NativeList<NDArray> opArrays = new NativeList<NDArray>(OP_ARRAY_SIZE, Allocator.Persistent);
    private const double EPSILON = 0.00000000001;
    private const int OP_ARRAY_SIZE = 100000;
    private static Random rng = new Random(GameManager.NDArrayGenSeed);
    private NativeArray<double> array;
    private Allocator allocator {
        get;
        set;
    }

    public NativeArray<int> shape {
        get;
        set;
    }

    public int numElements {
        get {
            int e = 1;
            for (int i = 0; i < shape.Length; i++) {
                e *= shape[i];
            }
            return e;
        }
    }

    public static NDArray NDArrayZeros(NativeArray<int> shape, Allocator allocator) {
        int numElem = 1;
        for (int i = 0; i < shape.Length; i++) {
            numElem *= shape[i];
        }
        NativeArray<double> data = new NativeArray<double>(numElem, allocator);
        for (int i = 0; i < numElem; i++) {
            data[i] = 0;
        }
        return new NDArray(shape, data, allocator);
    }

    public static NDArray RandomNDArray(NativeArray<int> shape, Allocator allocator) {
        int numElem = 1;
        for (int i = 0; i < shape.Length; i++) {
            numElem *= shape[i];
        }
        NativeArray<double> data = new NativeArray<double>(numElem, allocator);
        for (int i = 0; i < numElem; i++) {
            data[i] = rng.NextDouble();
        }
        return new NDArray(shape, data, allocator);
    }

    public NDArray (NativeArray<int> shape, NativeArray<double> data, Allocator allocator) {
        this.shape = shape;
        this.array = data;
        this.allocator = allocator;
    }

    public void Dispose() {
        array.Dispose();
        shape.Dispose();
    }

    public static void DisposeAll() {
        for (int i = 0; i < opArrays.Length; i++) {
            opArrays.ElementAt(i).Dispose();
        }
        opArrays.Clear();
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
            this.set(indices, value);
        }
    }

    public void set(int index, double value) {
        array[index] = value;
    }
    public void set(int[] indices, double value) {
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

    //Operations
    //NOTE: MUST DISPOSE OF RETURNED NDARRAY TO AVOID MEMORY LEAKS!!!!!

    public static NDArray operator > (NDArray a, int b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray greaterThan = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(greaterThan);
        for (int i = 0; i < a.numElements; i++) {
            greaterThan[i] = a[i] > b ? 1 : 0;
        }
        return greaterThan;
    }

    public static NDArray operator < (NDArray a, int b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray lessThan = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(lessThan);
        for (int i = 0; i < a.numElements; i++) {
            lessThan[i] = a[i] < b ? 1 : 0;
        }
        return lessThan;
    }

    public static NDArray operator > (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray greaterThan = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(greaterThan);
        for (int i = 0; i < a.numElements; i++) {
            greaterThan[i] = a[i] > b[i] ? 1 : 0;
        }
        return greaterThan;
    }

    public static NDArray operator < (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray lessThan = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(lessThan);
        for (int i = 0; i < a.numElements; i++) {
            lessThan[i] = a[i] < b[i] ? 1 : 0;
        }
        return lessThan;
    }

    public static NDArray operator * (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray product = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(product);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i]*b[i];
        }
        return product;
    }

    public static NDArray operator * (NDArray a, double b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray product = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(product);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i]*b;
        }
        return product;
    }

    public static NDArray operator * (double b, NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray product = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(product);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i]*b;
        }
        return product;
    }

    public static NDArray operator / (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray dividend = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(dividend);
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
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray dividend = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(dividend);
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
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray dividend = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(dividend);
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
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray sum = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(sum);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }
    public static NDArray operator + (double b, NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray sum = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(sum);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }

    public static NDArray operator + (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray sum = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(sum);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b[i];
        }
        return sum;
    }

    public static NDArray operator - (NDArray a, double b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray difference = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(difference);
        for (int i = 0; i < a.numElements; i++) {
            difference[i] = a[i] - b;
        }
        return difference;
    }
    public static NDArray operator - (double b, NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray difference = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(difference);
        for (int i = 0; i < a.numElements; i++) {
            difference[i] = b - a[i];
        }
        return difference;
    }

    public static NDArray operator - (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray difference = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(difference);
        for (int i = 0; i < a.numElements; i++) {
            difference[i] = a[i] - b[i];
        }
        return difference;
    }

    public static NDArray Exp (NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray exp = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(exp);
        for (int i = 0; i < a.numElements; i++) {
            exp[i] = math.exp(a[i]);
        }
        return exp;
    }

    public static NDArray Sqrt (NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray sqrt = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(sqrt);
        for (int i = 0; i < a.numElements; i++) {
            sqrt[i] = math.sqrt(a[i]);
        }
        return sqrt;
    }
    public static NDArray Dot (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(2, a.allocator);
        tempShape[0] = a.shape[0];
        tempShape[1] = b.shape[1];
        NDArray dot = new NDArray(tempShape, temp, a.allocator);
        NDArray.opArrays.Add(dot);
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
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray clamp = new NDArray(tempShape, temp, a.allocator);
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
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray clamp_back = new NDArray(tempShape, temp, a.allocator);
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
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray min = new NDArray(tempShape, temp, a.allocator);
        for (int i = 0; i < a.numElements; i++) {
            min[i] = math.min(a[i], b[i]);
        }
        return min;
    }

    public static NDArray Pow(NDArray a, double p) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray pow = new NDArray(tempShape, temp, a.allocator);
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
        NativeArray<double> temp = new NativeArray<double>(this.array.Length, allocator);
        NativeArray<int> tempShape = new NativeArray<int>(this.shape.Length, allocator);
        tempShape[0] = shape[1];
        tempShape[1] = shape[0];
        NDArray transpose = new NDArray(tempShape, temp, allocator);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                transpose[j, i] = this[i, j];
            }
        }
        return transpose;
    }

    public static NDArray Copy(NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray copy = new NDArray(tempShape, temp, a.allocator);
        for (int i = 0; i < a.numElements; i++) {
            copy[i] = a[i];
        }
        return copy;
    }
    public static NDArray Copy(NDArray a, Allocator allocator) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray copy = new NDArray(tempShape, temp, allocator);
        for (int i = 0; i < a.numElements; i++) {
            copy[i] = a[i];
        }
        return copy;
    }

    private int[] getStrides() {
        int mult = 1;
        int[] strides = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++) {
            strides[i] = shape[i] * mult;
            mult *= shape[i];
        }
        return strides;
    }
}
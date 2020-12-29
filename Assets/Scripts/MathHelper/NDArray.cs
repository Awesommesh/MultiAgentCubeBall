using Unity.Collections;
using Unity.Mathematics;
public struct NDArray {
    private const double EPSILON = 0.00000000001;
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

    public NDArray (NativeArray<int> shape, NativeArray<double> data, Allocator allocator) {
        this.shape = shape;
        this.array = data;
        this.allocator = allocator;
    }

    public void Dispose() {
        array.Dispose();
        shape.Dispose();
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
        for (int i = 0; i < a.numElements; i++) {
            lessThan[i] = a[i] > b ? 1 : 0;
        }
        return lessThan;
    }

    public static NDArray operator * (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray product = new NDArray(tempShape, temp, a.allocator);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i]*b[i];
        }
        return product;
    }

    public static NDArray operator / (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray dividend = new NDArray(tempShape, temp, a.allocator);
        for (int i = 0; i < a.numElements; i++) {
            if (b[i] == 0) {
                dividend[i] = a[i] / EPSILON;
            } else {
                dividend[i] = a[i]/b[i];
            }
        }
        return dividend;
    }

    public static NDArray operator + (NDArray a, double b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray sum = new NDArray(tempShape, temp, a.allocator);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }
    public static NDArray operator + (double b, NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NDArray sum = new NDArray(a.shape, temp, a.allocator);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }

    public static NDArray Exp (NDArray a) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(a.shape.Length, a.allocator);
        NativeArray<int>.Copy(a.shape, tempShape);
        NDArray exp = new NDArray(tempShape, temp, a.allocator);
        for (int i = 0; i < a.numElements; i++) {
            exp[i] = math.exp(a[i]);
        }
        return exp;
    }
    public static NDArray Dot (NDArray a, NDArray b) {
        NativeArray<double> temp = new NativeArray<double>(a.array.Length, a.allocator);
        NativeArray<int> tempShape = new NativeArray<int>(2, a.allocator);
        tempShape[0] = a.shape[0];
        tempShape[1] = b.shape[1];
        NDArray dot = new NDArray(tempShape, temp, a.allocator);
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

    public int[] getStrides() {
        int mult = 1;
        int[] strides = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++) {
            strides[i] = shape[i] * mult;
            mult *= shape[i];
        }
        return strides;
    }
}
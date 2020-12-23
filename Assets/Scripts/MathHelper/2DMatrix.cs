using UnityEngine;
using Unity.Collections;
using Unity.Entities;

public struct TwoDArray : IComponentData {
    const int MAX_SIZE = 10000;
    private DynamicBuffer<DoubleBufferElement> variable;
    private Entity var;

    public int numRow
    {
        get;
        set;
    }

    public int numCol {
        get;
        set;
    }

    public int numElements {
        get {
            return numRow*numCol;
        }
    }

    public TwoDArray(int numRow, int numCol) {
        this.numRow = numRow;
        this.numCol = numCol;
        EntityManager manager = World.DefaultGameObjectInjectionWorld.EntityManager;
        var = manager.CreateEntity();
        manager.AddBuffer<DoubleBufferElement>(var);
        variable = manager.GetBuffer<DoubleBufferElement>(var);
        variable.ResizeUninitialized(numRow * numCol);
    }

    public void Load(NativeArray<double> data) {
        NativeArray<DoubleBufferElement> bufferData = new NativeArray<DoubleBufferElement>();
        for (int i = 0; i < data.Length; i++) {
            bufferData[i] = new DoubleBufferElement(data[i]);
        }
        variable.CopyFrom(bufferData);
    }

    public void Fill(double val) {
        for (int i = 0; i < numElements; i++) {
            variable[i] = val;
        }
    }

    public double this[params int[] indices] {
        get {
            if (indices.Length == 1) {
                //Debug.Log("Here");
                //Debug.Log(indices[0]);
                return variable[indices[0]].Value;
            }
            //Debug.Log(indices.Length);
            return variable[indices[0]*numCol + indices[1]].Value;
        }
        set {
            if (indices.Length == 1) {
                //Debug.Log("Here");
                //Debug.Log(indices[0]);
                //Debug.Log(variable.Length);
                variable[indices[0]] = value;
            } else {
                variable[indices[0]*numCol + indices[1]] = value;
            }
        }
    }

    public static TwoDArray operator + (TwoDArray a, TwoDArray b) {
        TwoDArray sum = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b[i];
        }
        return sum;
    }

    public static TwoDArray operator + (double a, TwoDArray b) {
        TwoDArray sum = new TwoDArray(b.numRow, b.numCol);
        for (int i = 0; i < b.numElements; i++) {
            sum[i] = b[i] + a;
        }
        return sum;
    }
    public static TwoDArray operator + (TwoDArray a, double b) {
        TwoDArray sum = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }

    public static TwoDArray operator + (TwoDArray a, int b) {
        TwoDArray sum = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            sum[i] = a[i] + b;
        }
        return sum;
    }
    public static TwoDArray operator + (int a, TwoDArray b) {
        TwoDArray sum = new TwoDArray(b.numRow, b.numCol);
        for (int i = 0; i < b.numElements; i++) {
            sum[i] = a + b[i];
        }
        return sum;
    }

    public static TwoDArray operator - (TwoDArray a, TwoDArray b) {
        TwoDArray subtract = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            subtract[i] = a[i] - b[i];
        }
        return subtract;
    }

    public static TwoDArray operator * (TwoDArray a, TwoDArray b) {
        TwoDArray product = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            product[i] = a[i] * b[i];
        }
        return product;
    }
    public static TwoDArray operator / (TwoDArray a, TwoDArray b) {
        TwoDArray quotient = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            quotient[i] = a[i] / b[i];
        }
        return quotient;
    }

    public static TwoDArray operator > (TwoDArray a, TwoDArray b) {
        TwoDArray greaterThan = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            greaterThan[i] = a[i] > b[i] ? 1 : 0;
        }
        return greaterThan;
    }

    public static TwoDArray operator < (TwoDArray a, TwoDArray b) {
        TwoDArray lessThan = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            lessThan[i] = a[i] < b[i] ? 1 : 0;
        }
        return lessThan;
    }

    public static TwoDArray operator > (TwoDArray a, int b) {
        TwoDArray greaterThan = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            greaterThan[i] = a[i] > b ? 1 : 0;
        }
        return greaterThan;
    }
    public static TwoDArray operator < (TwoDArray a, int b) {
        TwoDArray lessThan = new TwoDArray(a.numRow, a.numCol);
        for (int i = 0; i < a.numElements; i++) {
            lessThan[i] = a[i] < b ? 1 : 0;
        }
        return lessThan;
    } 

    public TwoDArray Transpose() {
        return Operations.Transpose(this);
    }

    public TwoDArray appendOneCol() {
        TwoDArray appendedParam = new TwoDArray(numRow, numCol + 1);
        for (int i = 0; i < numRow; i++) {
            for (int j = 0; j < numCol; j++) {
                appendedParam[i, j] = this[i, j];
            }
            appendedParam[i, numCol] = 1;
        }
        return appendedParam;
    }

    public void Print()
        {
            string matrixString = "Matrix of dimensions " + numRow + "x" + numCol +":\n";
            for (int i = 0; i < numRow; i++)
            {
                for (int j = 0; j < numCol; j++)
                {
                    matrixString += (this[i, j] + "\t");
                }
                matrixString += "\n";
            }
            Debug.Log(matrixString);
        }
}

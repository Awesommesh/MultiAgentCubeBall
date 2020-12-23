using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Entities;

public struct TwoDArray : IComponentData {
    private NativeArray<double> variable;

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
        variable = new NativeArray<double>(numRow*numCol, Allocator.Persistent);
    }

    public void Load(NativeArray<double> data) {
        variable.CopyFrom(data);
    }

    public void Fill(double val) {
        for (int i = 0; i < numElements; i++) {
            variable[i] = val;
        }
    }

    public double this[params int[] indices] {
        get {
            if (indices.Length == 1) {
                return variable[indices[0]];
            }
            return variable[indices[0]*numCol + indices[1]];
        }
        set {
            variable[indices[0]*numCol + indices[1]] = value;
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
            for (int i = 0; i < numRow; i++)
            {
                for (int j = 0; j < numCol; j++)
                {
                    Debug.Log(this[i, j] + "\t");
                }
            }
        }
}

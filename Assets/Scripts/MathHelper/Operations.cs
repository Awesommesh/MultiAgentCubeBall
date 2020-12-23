using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Entities;

struct Operations : IComponentData {
    public static TwoDArray Random2DArray(int numRow, int numCol, uint seed) {
        TwoDArray random = new TwoDArray(numRow, numCol);
        Unity.Mathematics.Random rng = new Unity.Mathematics.Random(seed);
        for (int i = 0; i < random.numElements; i++) {
            random[i] = rng.NextDouble();
        }
        return random;
    }

    public static TwoDArray Dot (TwoDArray a, TwoDArray b) {
        if (a.numCol != b.numRow) {
            Debug.Log("Can't perform dot product due to mistmatch in dimensions");    
        }
        TwoDArray product = new TwoDArray(a.numRow, b.numCol);
        for (int i = 0; i < a.numRow; i++) {
            for (int j = 0; j < b.numCol; j++) {
                product[i, j] = 0;
                for (int k = 0; k < a.numCol; k++) {
                    product[i, j] += a[i, k] * b[k, j];
                }
            }
        }
        return product;
    }

    public static TwoDArray Exp (TwoDArray x) {
        TwoDArray exponent = new TwoDArray(x.numRow, x.numCol);
        for (int i = 0; i < x.numElements; i++) {
            exponent[i] = math.exp(x[i]);
        }
        return exponent;
    }

    public static TwoDArray Log (TwoDArray x) {
        TwoDArray log = new TwoDArray(x.numRow, x.numCol);
        for (int i = 0; i < x.numElements; i++) {
            log[i] = math.log(x[i]);
        }
        return log;
    } 
    public static TwoDArray Sqrt (TwoDArray x) {
        TwoDArray sqrt = new TwoDArray(x.numRow, x.numCol);
        for (int i = 0; i < x.numElements; i++) {
            sqrt[i] = math.sqrt(x[i]);
        }
        return sqrt;
    }

    public static TwoDArray Pow (TwoDArray x, double power) {
        TwoDArray pow = new TwoDArray(x.numRow, x.numCol);
        for (int i = 0; i < x.numElements; i++) {
            pow[i] = math.pow(x[i], power);
        }
        return pow;
    }

    public static TwoDArray Transpose (TwoDArray x) {
        TwoDArray transpose = new TwoDArray(x.numCol, x.numRow);
        for (int i = 0; i < x.numRow; i++) {
            for (int j = 0; j < x.numCol; j++) {
                transpose[j, i] = x[i, j];
            }
        }
        return transpose;
    }
}
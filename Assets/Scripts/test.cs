using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;
public class test : MonoBehaviour
{
    // Start is called before the first frame update
    unsafe void Start()
    {
        NativeArray<int> aShape = new NativeArray<int>(2, Allocator.Persistent);
        aShape[0]=3;
        aShape[1]=2;
        NativeArray<double> aData = new NativeArray<double>(6, Allocator.Persistent);
        aData[0] = 1;
        aData[1] = 7;
        aData[2] = 3;
        aData[3] = 4;
        aData[4] = 5;
        aData[5] = 6;



        /*NativeArray<double> aOnes = new NativeArray<double>(9, Allocator.Persistent);
        NativeArray<int> aOnesShape = new NativeArray<int>(2, Allocator.Persistent);
        aOnesShape[0]=3;
        aOnesShape[1]=3;

        NativeNDOps.appendOnes(ref aOnes, aData, 3, 2);
        Debug.Log("aOnes: ");
        printMatrix(aOnesShape, aOnes);*/
        
        NativeArray<int> bShape = new NativeArray<int>(2, Allocator.Persistent);
        bShape[0]=3;
        bShape[1]=2;
        NativeArray<double> bData = new NativeArray<double>(6, Allocator.Persistent);
        bData[0] = 1;
        bData[1] = 2;
        bData[2] = 3;
        bData[3] = 4;
        bData[4] = 5;
        bData[5] = 6;



        NativeArray<int> outputShape = new NativeArray<int>(2, Allocator.Persistent);
        outputShape[0] = 2;
        outputShape[1] = 2;
        NativeArray<double> output= new NativeArray<double>(4, Allocator.Persistent);

        for (int i = 0; i < 4; i++) {
            output[i] = 0;
        }

        Debug.Log("a: ");
        printMatrix(aShape, aData);

        Debug.Log("b: ");
        printMatrix(bShape, bData);

        Debug.Log("Dot Product: ");
        printMatrix(outputShape, output);

        NativeArray<double> test = new NativeArray<double>(3, Allocator.Persistent);
        Debug.Log("Partial MemCpy");
        NativeNDOps.CopyPartial(aData, test, 3);
        Debug.Log(test[0] + " "  +  test[1] + " " + test[2]);

        NativeArray<int> functionOutShape = new NativeArray<int>(2, Allocator.Persistent);
        functionOutShape[0] = 2;
        functionOutShape[1] = 3;
        NativeArray<double> functionOut = new NativeArray<double>(6, Allocator.Persistent);


        Debug.Log("FeedFwd function output tests");

        NativeNDOps.ActivationFunction(ActivationType.Sigmoid, aData, functionOut);
        Debug.Log("Sigmoid of a: ");
        printMatrix(functionOutShape, functionOut);

        NativeNDOps.ActivationFunction(ActivationType.ReLU, aData, functionOut);
        Debug.Log("ReLU of a: ");
        printMatrix(functionOutShape, functionOut);

        NativeNDOps.ActivationFunction(ActivationType.None, aData, functionOut);
        Debug.Log("Default of a: ");
        printMatrix(functionOutShape, functionOut);

        Debug.Log("BackProp function output tests");

        NativeArray<double> gradient = new NativeArray<double>(6, Allocator.Persistent);
        for (var i=0; i<gradient.Length; i++){
            gradient[i] = 1.0;
        }

        NativeNDOps.ActivationFunctionBack(ActivationType.Sigmoid, aData, gradient, functionOut);
        Debug.Log("Sigmoid of a: ");
        printMatrix(functionOutShape, functionOut);

        NativeNDOps.ActivationFunctionBack(ActivationType.ReLU, aData, gradient, functionOut);
        Debug.Log("ReLU of a: ");
        printMatrix(functionOutShape, functionOut);

        NativeNDOps.ActivationFunctionBack(ActivationType.None, aData, gradient, functionOut);
        Debug.Log("Default of a: ");
        printMatrix(functionOutShape, functionOut);

        aShape.Dispose();
        aData.Dispose();
        bShape.Dispose();
        bData.Dispose();
        outputShape.Dispose();
        output.Dispose();
        functionOut.Dispose();
        functionOutShape.Dispose();
        gradient.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    [DllImport("/Users/animeshagrawal/repositories/MultiAgentCubeBall/Assets/Scripts/MathHelper/matmul.dylib")]
    public static extern unsafe void matmul(int transA, int transB, int A_width, int A_height, int B_width, int B_height, 
        double* A, double* B, double* output);

    void printMatrix(NativeArray<int> aShape, NativeArray<double> a){
        string str = "";
        for (var i = 0; i<aShape[0]; i++){
            for (var j=0; j<aShape[1]; j++){
                str += a[i*aShape[1]+j] + "\t";
            }
            str += "\n";
        }
        Debug.Log(str);
    }
}

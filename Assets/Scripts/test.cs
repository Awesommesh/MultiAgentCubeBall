using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
public class test : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        NativeArray<int> aShape = new NativeArray<int>(2, Allocator.Persistent);
        aShape[0]=2;
        aShape[1]=3;
        NativeArray<double> aData = new NativeArray<double>(6, Allocator.Persistent);
        aData[0] = -1.0;
        aData[1] = -2.0;
        aData[2] = -3.0;
        aData[3] = 0.0;
        aData[4] = 5.0;
        aData[5] = 6.0;
        
        NativeArray<int> bShape = new NativeArray<int>(2, Allocator.Persistent);
        bShape[0]=3;
        bShape[1]=2;
        NativeArray<double> bData = new NativeArray<double>(6, Allocator.Persistent);
        bData[0] = 1.0;
        bData[1] = 2.0;
        bData[2] = 3.0;
        bData[3] = 4.0;
        bData[4] = 5.0;
        bData[5] = 6.0;

        NativeArray<int> outputShape = new NativeArray<int>(2, Allocator.Persistent);
        outputShape[0] = 2;
        outputShape[1] = 2;
        NativeArray<double> output= new NativeArray<double>(4, Allocator.Persistent);
    
        Debug.Log("a: ");
        printMatrix(aShape, aData);

        Debug.Log("b: ");
        printMatrix(bShape, bData);

        //NativeNDOps.Dot(aData, aShape, bData, bShape, output);

        Debug.Log("Dot Product: ");
        printMatrix(outputShape, output);

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

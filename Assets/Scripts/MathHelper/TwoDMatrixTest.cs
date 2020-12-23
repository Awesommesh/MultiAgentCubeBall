using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TwoDMatrixTest : MonoBehaviour
{
    // Start is called before the first frame update
    public TwoDArray array;
    void Start()
    {
        array = Operations.Random2DArray(5, 5, 42);
        array.Print();
    }
}

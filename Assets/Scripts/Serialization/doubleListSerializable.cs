using System.Collections.Generic;
using Unity.Collections;

[System.Serializable]
public class doubleListSerializable
{	
    public double[] data;
    public int Length {
        get {
            return this.data.Length;
        }
    }
	public doubleListSerializable (List<double> input){
        this.data = new double[input.Count];
        for (int i=0; i<input.Count; i++){
            this.data[i] = input[i];
        }
	}

    public doubleListSerializable (NativeArray<double> input){
        this.data = new double[input.Length];
        for (int i=0; i<input.Length; i++){
            this.data[i] = input[i];
        }
	}

    public double this[int index] {
        get {
            return data[index];
        }
        set {
            data[index] = value;
        }
    }
}

using System.Collections.Generic;
using Unity.Collections;

[System.Serializable]
public class intListSerializable
{	
    public int[] data;
    public int Length {
        get {
            return this.data.Length;
        }
    }
	public intListSerializable (List<int> input){
        this.data = new int[input.Count];
        for (int i=0; i<input.Count; i++){
            this.data[i] = input[i];
        }
	}

    public intListSerializable (NativeArray<int> input){
        this.data = new int[input.Length];
        for (int i=0; i<input.Length; i++){
            this.data[i] = input[i];
        }
	}

    public int this[int index] {
        get {
            return data[index];
        }
        set {
            data[index] = value;
        }
    }
}

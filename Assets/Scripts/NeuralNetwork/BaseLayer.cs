using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Entities;
public struct BaseLayer : IComponentData {
    public string Name {
        get;
        set;
    }

    public TwoDArray Input {
        get;
        set;
    }

    public TwoDArray Output {
        get;
        set;
    }

    public TwoDArray Parameters {
        get;
        set;
    }

    public BaseLayer(string name, TwoDArray Input, TwoDArray Parameters) {
        this.Name = name;
        this.Input = Input;
        this.Parameters = Parameters;
        this.Output = new TwoDArray(Input.numRow, Parameters.numCol);
    }

    

}

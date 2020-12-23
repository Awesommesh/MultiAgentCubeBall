using UnityEngine;
using Unity.Collections;
using Unity.Entities;

public struct DoubleBufferElement : IBufferElementData {
    public double Value {
        get;
        set;
    }

    public static implicit operator DoubleBufferElement(double e) {
        return new DoubleBufferElement{Value = e};
    }

    public DoubleBufferElement(double e) {
        this.Value = e;
    }
}
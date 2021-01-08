using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NNSerializableWrapper : MonoBehaviour
{	
	public NNSerializable[] actors;
	public NNSerializable[] critics;

	public NNSerializableWrapper(NNSerializable[] actors, NNSerializable[] critics){
		this.actors = actors;
		this.critics = critics;
	}
}

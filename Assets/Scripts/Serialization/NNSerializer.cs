using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NNSerializer
{	
	public static void Serialize(string path, NeuralNetwork[] actors, NeuralNetwork[] critics){
		NNSerializable[] serializeableActors = new NNSerializable[actors.Length];
		NNSerializable[] serializeableCritics = new NNSerializable[critics.Length];
		for(int i=0; i<actors.Length; i++){
			serializeableActors[i] = new NNSerializable(actors[i]);
			serializeableCritics[i] = new NNSerializable(critics[i]);
		}
		NNSerializableWrapper jsonObject = new NNSerializableWrapper(serializeableActors, serializeableCritics);
		HandleTextFile.WriteString(path, JsonUtility.ToJson(jsonObject));
	}
	public static void Deserialize(string path, NeuralNetwork[] actors, NeuralNetwork[] critics){
		string txt = HandleTextFile.ReadString(path);
		NNSerializableWrapper data = JsonUtility.FromJson<NNSerializableWrapper>(txt);
		for (int i=0; i<data.actors.Length; i++){
			actors[i] = NeuralNetwork.SerializableToNN(data.actors[i], GameManager.ACTOR_LR);
			critics[i] = NeuralNetwork.SerializableToNN(data.critics[i], GameManager.CRITIC_LR);
		}
	}
}
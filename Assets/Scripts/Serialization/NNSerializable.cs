using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NNSerializable
{	
	public intListSerializable[] weightsShape;
	public doubleListSerializable[] weights;
	public int numLayers;
	public int[] activations;
	public int numInputs;
	public int numOutputs;
	public double[] stdVals;
	public NNSerializable(NeuralNetwork nn){
		weightsShape = new intListSerializable[nn.weightsShape.Length];
		weights = new doubleListSerializable[nn.weights.Length];
		stdVals = new double[nn.std.Length];
		activations = new int[nn.activations.Length];
		for (int i = 0; i < nn.numLayers; i++) {
			weightsShape[i] = new intListSerializable(nn.weightsShape[i]);
			weights[i] = new doubleListSerializable(nn.weights[i]);
		}
			
		for (int i=0; i<nn.std.Length; i++){
			stdVals[i] = nn.std[i];
		}

		for (int i=0; i<nn.activations.Length; i++){
			activations[i] = (int)nn.activations[i];
		}
		numLayers = nn.numLayers;
		numInputs = nn.numInputs;
		numOutputs = nn.numOutputs;

	}
}

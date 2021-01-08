using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NNSerializable : MonoBehaviour
{	
	public int[][] weightsShape;
	public double[][] weights;
	public int numLayers;
	public int[] activations;
	public int numInputs;
	public int numOutputs;
	public double[] stdVals;
	public NNSerializable(NeuralNetwork nn){
		weightsShape = new int[nn.weightsShape.Length][];
		weights = new double[nn.weights.Length][];

		for (int i=0; i<nn.weightsShape.Length; i++){
			weightsShape[i] = new int[nn.weightsShape[i].Length];
			for (int j=0; j<nn.weightsShape[i].Length; j++){
				weightsShape[i][j] = nn.weightsShape[i][j];
			}
		}
		for (int i=0; i<nn.weights.Length; i++){
			weights[i] = new double[nn.weights[i].Length];
			for (int j=0; j<nn.weights[i].Length; j++){
				weights[i][j] = nn.weights[i][j];
			}
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

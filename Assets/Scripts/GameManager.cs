using Unity.Mathematics;
using UnityEngine;
using Unity.Collections;

public class GameManager : MonoBehaviour
{
    public static uint SEED = 1;
    public static uint NDArrayGenSeed = 2;
    public static double GAE_LAMBDA = 0.95;
    public static double GAMMA = 0.99;
    public static double ALPHA = 0.001;
    public static double BETA1 = 0.9;
    public static double BETA2 = 0.999;
    public static double EPSILON = 0.00000001;
    public int numLayers;
    public int STATE_SIZE = 70;
    public int numOutputs;
    public NeuralNetwork[] agents;
    public NeuralNetwork[] critics;
    public int EPISODE_LENGTH = 256;
    public int episode_iteration = 0;
    public GameObject[] blueTeam;
    public GameObject[] redTeam;
    public GameObject blueGoal;
    public GameObject redGoal;
    public GameObject ball;
    void Awake() {
        episode_iteration = 0;
        NativeArray<ActivationType> activationList = new NativeArray<ActivationType>(numLayers, Allocator.Persistent);
        activationList[0] = ActivationType.ReLU;
        activationList[1] = ActivationType.ReLU;
        activationList[2] = ActivationType.ReLU;
        activationList[3] = ActivationType.ReLU;
        activationList[4] = ActivationType.None;
        NativeArray<int>[] layerShapes = new NativeArray<int>[numLayers];
        for (int i = 0; i < layerShapes.Length; i++) {
            layerShapes[i] = new NativeArray<int>(2, Allocator.Persistent);
        }
        //Hidden Layer 1: 70 inputs, 210 outputs;
        layerShapes[0][0] = 210;
        layerShapes[0][1] = 70;
        //Hidden Layer 2: 210 inputs, 630 outputs;
        layerShapes[1][0] = 630;
        layerShapes[1][1] = 210;
        //Hidden Layer 3: 630 inputs, 315 outputs;
        layerShapes[2][0] = 315;
        layerShapes[2][1] = 630;
        //Hidden Layer 4: 315 inputs, 63 outputs;
        layerShapes[3][0] = 63;
        layerShapes[3][1] = 315;
        //Hidden Layer 5: 63 inputs, 2 outputs;
        layerShapes[0][0] = 2;
        layerShapes[0][1] = 63;
        for (int i = 0; i < agents.Length; i++) {
            NativeArray<NDArray> initialWeights = new NativeArray<NDArray>(numLayers, Allocator.Persistent);
            for (int j = 0; j < layerShapes.Length; j++) {
                initialWeights[i] = NDArray.RandomNDArray(layerShapes[i], Allocator.Persistent);
            }
            NativeArray<double> stdVals = new NativeArray<double>(numOutputs, Allocator.Persistent);
            for (int j = 0; j < stdVals.Length; j++) {
                stdVals[j] = 0;
            }
            agents[i] = new NeuralNetwork(numLayers, activationList, initialWeights, STATE_SIZE, numOutputs, stdVals, ALPHA, BETA1, BETA2, EPSILON);
            critics[i] = new NeuralNetwork(numLayers, activationList, initialWeights, STATE_SIZE, 1, stdVals, ALPHA, BETA1, BETA2, EPSILON);
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (episode_iteration < 256) {

            episode_iteration++;
        }
    }

    public void DisposeAll() {
        for (int i = 0; i < agents.Length; i++) {
            agents[i].Dispose();
            critics[i].Dispose();
        }
    }
}

using Unity.Mathematics;
using UnityEngine;
using Unity.Collections;

public class GameManager : MonoBehaviour
{
    public static uint SEED = 1;
    public static uint NDArrayGenSeed = 2;
    public int NUM_ENV = 1;
    public static double GAE_LAMBDA = 0.95;
    public static double GAMMA = 0.99;
    public static double ALPHA = 0.001;
    public static double BETA1 = 0.9;
    public static double BETA2 = 0.999;
    public static double EPSILON = 0.00000001;
    public static int EPISODE_LENGTH = 256;
    public int MINI_BATCH_SIZE = 64;
    public int PPO_EPOCHS = 10;
    public static int TEAM_SIZE = 5;
    public static double FIELD_LENGTH = 90;
    public static double MAX_SPEED = 5;
    public int numLayers;
    public static int STATE_SIZE = 70;
    public static int NUM_OUTPUTS = 3;
    public static float PHYSICS_STEP_SIZE = 0.01f;
    public const float X_Env_Increment = 112.5f;
    public const float Z_Env_Increment = 60f;
    public NeuralNetwork[] agents;
    public NeuralNetwork[] critics;
    
    public int episode_iteration = 0;
    public GameObject gameEnv;
    public GameObject gameManager;
    //public Experience[] envs;
    /*void Awake() {
        Physics.autoSimulation = false;
        agents = new NeuralNetwork[TEAM_SIZE];
        critics = new NeuralNetwork[TEAM_SIZE];
        float sqrt = math.sqrt(NUM_ENV);
        if (sqrt-((int)sqrt) > 0) {
            sqrt++;
        }
        int gridLength = (int)math.sqrt(NUM_ENV);
        int numEnvCreated = 0;
        //Setup parallel environments (for now only works with one environment)
        //envs = new Experience[NUM_ENV];
        for (int i = 0; i < gridLength; i++) {
            //envs[i] = new Experience();
            for (int j = 0; j < gridLength; j++) {
                GameObject newEnv = Instantiate(gameEnv);
                newEnv.transform.Translate(new Vector3(i*X_Env_Increment, 0.5f, j*Z_Env_Increment), Space.World);
                if (numEnvCreated >= NUM_ENV) {
                    break;
                }
                numEnvCreated++;
            }
        }

        episode_iteration = 0;

        //Initialize NN randomly with correct architecture etc.
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
            NativeHashMap<int, NDArray> initialWeights = new NativeHashMap<int, NDArray>(numLayers, Allocator.Persistent);
            for (int j = 0; j < layerShapes.Length; j++) {
                initialWeights.Add(j, NDArray.RandomNDArray(layerShapes[j], Allocator.Persistent));
            }
            NativeArray<double> stdVals = new NativeArray<double>(NUM_OUTPUTS, Allocator.Persistent);
            for (int j = 0; j < stdVals.Length; j++) {
                stdVals[j] = 0;
            }
            agents[i] = new NeuralNetwork(numLayers, activationList, initialWeights, STATE_SIZE, NUM_OUTPUTS, stdVals, ALPHA, BETA1, BETA2, EPSILON);
            critics[i] = new NeuralNetwork(numLayers, activationList, initialWeights, STATE_SIZE, 1, stdVals, ALPHA, BETA1, BETA2, EPSILON);
        }
    }*/

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (episode_iteration < 256) {
            //Perform Blue Team Actions
            
            //Perform Red Team Actions
            episode_iteration++;
        }
        if (episode_iteration == 256) {

        }
        Physics.Simulate(PHYSICS_STEP_SIZE);
    }

    void Reset() {

    }
}

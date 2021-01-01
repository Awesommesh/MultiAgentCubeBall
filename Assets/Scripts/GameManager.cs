using Unity.Mathematics;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;

public class GameManager : MonoBehaviour
{
    public static uint SEED = 1;
    public static uint NDArrayGenSeed = 2;
    public uint MINI_BATCH_SEED = 0;
    public int NUM_ENV = 1;
    public static double GAE_LAMBDA = 0.95;
    public static double GAMMA = 0.99;
    public static double ALPHA = 0.001;
    public static double BETA1 = 0.9;
    public static double BETA2 = 0.999;
    public static double EPSILON = 0.00000001;
    public static int EPISODE_LENGTH = 256;
    public static int BATCH_SIZE;
    public int MINI_BATCH_SIZE = 64;
    public int PPO_EPOCHS = 10;
    public static int TEAM_SIZE = 5;
    public static double FIELD_LENGTH = 90;
    public static double MAX_SPEED = 5;
    public int numLayers;
    public static int STATE_SIZE = 70;
    public const double PPO_EPILSON = 0.2;
    public const double CRITIC_DISCOUNT = 0.5;
    public const double ENTROPY_BETA = 0.001;
    //Ou
    public static int NUM_ACTIONS = 3;
    public static float PHYSICS_STEP_SIZE = 0.01f;
    public const float X_Env_Increment = 112.5f;
    public const float Z_Env_Increment = 60f;
    public NeuralNetwork[] agents;
    public NeuralNetwork[] critics;
    
    public int episode_iteration = 0;
    public GameObject gameEnv;
    public GameObject gameManager;

    //PPO Experience Accumulation
    private NativeArray<double> states;
    private NativeArray<double> actions;
    private NativeArray<double> log_probs;
    private NativeArray<double> returns;
    private NativeArray<double> advantages;
    private NativeArray<double> actorGrads;
    private NativeArray<double> criticGrads;

    private NativeArray<int> shuffle;
    private NativeArray<int> minibatches;

    //public Experience[] envs;
    void Awake() {
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

        BATCH_SIZE = EPISODE_LENGTH * NUM_ENV * TEAM_SIZE * 2;
        minibatches = new NativeArray<int>(BATCH_SIZE, Allocator.Persistent);
        shuffle = new NativeArray<int>(BATCH_SIZE, Allocator.Persistent);
        for (int i = 0; i < shuffle.Length; i++) {
            shuffle[i] = i;
        }
        states = new NativeArray<double>(BATCH_SIZE*STATE_SIZE, Allocator.Persistent);
        actions = new NativeArray<double>(BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
        log_probs = new NativeArray<double>(BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
        actorGrads = new NativeArray<double>(BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
        returns = new NativeArray<double>(BATCH_SIZE, Allocator.Persistent);
        advantages = new NativeArray<double>(BATCH_SIZE, Allocator.Persistent);
        criticGrads = new NativeArray<double>(BATCH_SIZE, Allocator.Persistent);
        episode_iteration = 0;

        /*//Initialize NN randomly with correct architecture etc.
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
            NativeArray<double> stdVals = new NativeArray<double>(NUM_ACTIONS, Allocator.Persistent);
            for (int j = 0; j < stdVals.Length; j++) {
                stdVals[j] = 0;
            }
            agents[i] = new NeuralNetwork(numLayers, activationList, initialWeights, STATE_SIZE, NUM_ACTIONS, stdVals, ALPHA, BETA1, BETA2, EPSILON);
            critics[i] = new NeuralNetwork(numLayers, activationList, initialWeights, STATE_SIZE, 1, stdVals, ALPHA, BETA1, BETA2, EPSILON);
        }
        */
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (episode_iteration < EPISODE_LENGTH) {
            //Perform Blue Team Actions
            
            //Perform Red Team Actions
            episode_iteration++;
        }
        if (episode_iteration == EPISODE_LENGTH) {
            PPO_Update();
        }
        Physics.Simulate(PHYSICS_STEP_SIZE);
    }

    void PPO_Update() {
        int numMiniBatches = BATCH_SIZE / MINI_BATCH_SIZE;
        GenerateMiniBatchesJob batchJob = new GenerateMiniBatchesJob {
            seed = MINI_BATCH_SEED,
            MINI_BATCH_SIZE = MINI_BATCH_SIZE,
            minibatches = minibatches,
            shuffle = shuffle
        };
        JobHandle jobHandle = batchJob.Schedule(numMiniBatches, 4);
        jobHandle.Complete();

        //PPO
        //Perform forward on each agent for each transition in batch
        NativeArray<double> actionDists = new NativeArray<double>(minibatches.Length*NUM_ACTIONS, Allocator.TempJob);
        NativeArray<double> stateVals = new NativeArray<double>(minibatches.Length, Allocator.TempJob);
        for (int i = 0; i < TEAM_SIZE; i++) {
            //Perform forward network on every single transition
            int actionInd = 0;
            int stateValInd = 0;
            for (int j = 0; j < states.Length; j+= STATE_SIZE) {
                NDArray curActionDist = agents[i].Forward(NDArray.fromNativeArray(states, j, STATE_SIZE));
                curActionDist.fillNativeArray(actionDists, actionInd, NUM_ACTIONS);
                stateVals[stateValInd] = (agents[i].Forward(NDArray.fromNativeArray(states, j, STATE_SIZE)))[0];
                actionInd+=NUM_ACTIONS;
                stateValInd++;
            }
        }

        //For Each agent, for each minibatch, perform PPO update step
        for (int i  = 0; i < TEAM_SIZE; i++) {
            for (int j = 0; j < numMiniBatches; j++) {
                NativeArray<double> nativeActorGrads = new NativeArray<double>(MINI_BATCH_SIZE * NUM_ACTIONS, Allocator.Persistent);
                NativeArray<double> nativeCriticGrads = new NativeArray<double>(MINI_BATCH_SIZE, Allocator.Persistent);
                PPOUpdateJob ppoJob = new PPOUpdateJob {
                    minibatches = minibatches,
                    batchInd = j,
                    actionDists = actionDists,
                    stateVals = stateVals,
                    actions = actions,
                    old_log_probs = log_probs,
                    returns = returns,
                    advantages = advantages,
                    log_std = agents[i].log_std,
                    log_std_mean = agents[i].log_std_mean,
                    NUM_ACTIONS = NUM_ACTIONS,
                    PPO_EPILSON = PPO_EPILSON,
                    CRITIC_DISCOUNT = CRITIC_DISCOUNT,
                    ENTROPY_BETA = ENTROPY_BETA,
                    MINI_BATCH_SIZE = MINI_BATCH_SIZE,
                    actorGrad = nativeActorGrads,
                    criticGrad = nativeCriticGrads
                };
                JobHandle ppoJobHandle = ppoJob.Schedule();
                ppoJobHandle.Complete();

                for (int k = 0; k < MINI_BATCH_SIZE; k++) {
                    NDArray actorGrads = NDArray.fromNativeArray(nativeActorGrads, k * NUM_ACTIONS, NUM_ACTIONS);
                    agents[i].Backward(actorGrads);
                    NDArray criticGrads = NDArray.fromNativeArray(nativeCriticGrads, k, 1);
                    critics[i].Backward(criticGrads);
                }
            }
            
        }
    }

    void Reset() {

    }
}

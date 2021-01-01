using Unity.Mathematics;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;

public class GameManager : MonoBehaviour
{
    public static uint SEED = 1;
    public static uint NDArrayGenSeed = 2;
    public uint MINI_BATCH_SEED = 4;
    public int NUM_ENV = 1;
    public static double GAE_LAMBDA = 0.95;
    public static double GAMMA = 0.99;
    public static double ALPHA = 0.001;
    public static double BETA1 = 0.9;
    public static double BETA2 = 0.999;
    public static double EPSILON = 0.00000001;
    public static int EPISODE_LENGTH = 64;
    public static int BATCH_SIZE;
    public int MINI_BATCH_SIZE = 16;
    public int PPO_EPOCHS = 10;
    public static int TEAM_SIZE = 5;
    public static double FIELD_LENGTH = 90;
    public static double MAX_SPEED = 50;
    public int numLayers;
    public static int STATE_SIZE = 76;
    public const double PPO_EPILSON = 0.2;
    public const double CRITIC_DISCOUNT = 0.5;
    public const double ENTROPY_BETA = 0.001;
    public int interval = 4;
    public static int NUM_ACTIONS = 3;
    public static float PHYSICS_STEP_SIZE = 0.01f;
    public const float X_Env_Increment = 112.5f;
    public const float Z_Env_Increment = 60f;
    public NeuralNetwork[] agents;
    public NeuralNetwork[] critics;
    public int[,] layerShapes;
    public double[] actorLog_Std;
    public double[] criticLog_Std;
    public int episode_iteration = 0;
    public GameObject gameEnv;

    //PPO Experience Accumulation
    private NativeArray<double> states;
    private NativeArray<double> actions;
    private NativeArray<double> log_probs;
    private NativeArray<double> returns;
    private NativeArray<double> advantages;

    private NativeArray<int> shuffle;
    private NativeArray<int> minibatches;

    public Experience[] envs;
    void Awake() {
        Physics.autoSimulation = false;
        layerShapes = new int[numLayers,2];
        //Have to hard code layer shapes ...
        //layer 1
        layerShapes[0, 0] = 228;
        layerShapes[0, 1] = STATE_SIZE;
        //layer 2
        layerShapes[1, 0] = 684;
        layerShapes[1, 1] = 228;
        //layer 3
        layerShapes[2, 0] = 342;
        layerShapes[2, 1] = 684;
        //layer 4
        layerShapes[3, 0] = 171;
        layerShapes[3, 1] = 342;
        //layer 5
        layerShapes[4, 0] = 3;
        layerShapes[4, 1] = 171;

        actorLog_Std = new double[NUM_ACTIONS];
        for (int i = 0; i < NUM_ACTIONS; i++) {
            actorLog_Std[i] = 0;
        }
        criticLog_Std = new double[1];
        criticLog_Std[0] = 0;
        agents = new NeuralNetwork[TEAM_SIZE];
        critics = new NeuralNetwork[TEAM_SIZE];
        float sqrt = math.sqrt(NUM_ENV);
        if (sqrt-((int)sqrt) > 0) {
            sqrt++;
        }
        int gridLength = (int)math.sqrt(NUM_ENV);
        int numEnvCreated = 0;
        //Setup parallel environments (for now only works with one environment)
        envs = new Experience[NUM_ENV];
        for (int i = 0; i < gridLength; i++) {
            for (int j = 0; j < gridLength; j++) {
                if (numEnvCreated >= NUM_ENV) {
                    break;
                }
                GameObject newEnv = Instantiate(gameEnv);
                newEnv.transform.Translate(new Vector3(i*X_Env_Increment, 0.5f, j*Z_Env_Increment), Space.World);
                GameObject newBall = newEnv.transform.Find("Ball").gameObject;
                GameObject newBlueGoal = newEnv.transform.Find("GoalAreaBlue").gameObject;
                GameObject newRedGoal = newEnv.transform.Find("GoalAreaRed").gameObject;
                GameObject[] newBlueTeam = new GameObject[TEAM_SIZE];
                GameObject[] newRedTeam = new GameObject[TEAM_SIZE];
                for (int k = 0; k < TEAM_SIZE; k++) {
                    newBlueTeam[k] = newEnv.transform.Find("BlueTeam/Blue"+(k+1)).gameObject;
                    newRedTeam[k] = newEnv.transform.Find("RedTeam/Red"+(k+1)).gameObject;
                }
                envs[i] = new Experience(newBall, newBlueGoal, newRedGoal, newBlueTeam, newRedTeam);
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
        returns = new NativeArray<double>(BATCH_SIZE, Allocator.Persistent);
        advantages = new NativeArray<double>(BATCH_SIZE, Allocator.Persistent);
        episode_iteration = 0;

        //Initialize NN randomly with correct architecture etc.
        ActivationType[] activationList = new ActivationType[numLayers];
        activationList[0] = ActivationType.ReLU;
        activationList[1] = ActivationType.ReLU;
        activationList[2] = ActivationType.ReLU;
        activationList[3] = ActivationType.ReLU;
        activationList[4] = ActivationType.None;
        
        
        for (int i = 0; i < agents.Length; i++) {
            NDArray[] weights = new NDArray[numLayers];
            for (int j = 0; j < numLayers; j++) {
                int[]curLayerShape = new int[2];
                curLayerShape[0] = layerShapes[j, 0];
                curLayerShape[1] = layerShapes[j, 1];
                weights[j] = NDArray.HeInitializedNDArray(curLayerShape, layerShapes[j, 1]);
            }
            agents[i] = new NeuralNetwork(numLayers, activationList, weights, STATE_SIZE, NUM_ACTIONS, actorLog_Std, ALPHA, BETA1, BETA2, EPSILON);
            weights = new NDArray[numLayers];
            for (int j = 0; j < numLayers; j++) {
                int[]curLayerShape = new int[2];
                if (j == numLayers - 1) {
                    curLayerShape[0] = 1;
                } else {
                    curLayerShape[0] = layerShapes[j, 0];
                }
                
                curLayerShape[1] = layerShapes[j, 1];
                weights[j] = NDArray.HeInitializedNDArray(curLayerShape, layerShapes[j, 1]);
            }
            critics[i] = new NeuralNetwork(numLayers, activationList, weights, STATE_SIZE, 1, criticLog_Std, ALPHA, BETA1, BETA2, EPSILON);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Time.frameCount % interval == 0) {
            if (episode_iteration < EPISODE_LENGTH) {
                for (int i = 0; i < NUM_ENV; i++) {
                    envs[i].stepForward(agents, critics, agents, critics);
                }
                episode_iteration++;
            } else if (episode_iteration == EPISODE_LENGTH) {
                for (int i = 0; i < NUM_ENV; i++) {
                    envs[i].getNextValues(critics, critics);
                    envs[i].CalculateGAE();
                    //Merge current environment data with global experience data
                    //For each player on given env append all of their experiences to global experience pool
                    int globalInd = 0;
                    for (int j = 0; j < TEAM_SIZE; j++) {
                        for (int k = 0; k < EPISODE_LENGTH; k++) {
                            //Blue and Red Agent j's kth time_step
                            int nextGlobalInd = globalInd + 1;
                            for (int l = 0; l < STATE_SIZE; l++) {
                                states[globalInd*STATE_SIZE + l] = envs[i].blueStates[k, l];
                                states[nextGlobalInd*STATE_SIZE + l] = envs[i].redStates[k, l];
                            }
                            for (int l = 0; l < NUM_ACTIONS; l++) {
                                actions[globalInd*NUM_ACTIONS + l] = envs[i].blueActions[k, j, l];
                                log_probs[globalInd*NUM_ACTIONS + l] = envs[i].blueLog_Probs[k, j, l];
                                actions[nextGlobalInd*NUM_ACTIONS + l] = envs[i].redActions[k, j, l];
                                log_probs[nextGlobalInd*NUM_ACTIONS + l] = envs[i].redLog_Probs[k, j, l];
                            }
                            returns[globalInd] = envs[i].blueReturns[k*TEAM_SIZE + j];
                            advantages[globalInd] = envs[i].blueAdvantages[k*TEAM_SIZE + j];
                            returns[nextGlobalInd] = envs[i].redReturns[k*TEAM_SIZE + j];
                            advantages[nextGlobalInd] = envs[i].redAdvantages[k*TEAM_SIZE + j];
                            globalInd+=2;
                        }
                    }
                }
                
                //for (int i = 0; i < PPO_EPOCHS; i++) {
                PPO_Update();
                Debug.Log("herte");
                //}
                //Need to reset environments;
                //episode_iteration = 0;
            }
            Physics.Simulate(PHYSICS_STEP_SIZE);
        }
        
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

        Debug.Log("mini batch job done");

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

        Debug.Log("did forward update");

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

                actionDists.Dispose();
                stateVals.Dispose();

                for (int k = 0; k < MINI_BATCH_SIZE; k++) {
                    NDArray actorGrads = NDArray.fromNativeArray(nativeActorGrads, k * NUM_ACTIONS, NUM_ACTIONS);
                    agents[i].Backward(actorGrads);
                    NDArray criticGrads = NDArray.fromNativeArray(nativeCriticGrads, k, 1);
                    critics[i].Backward(criticGrads);
                }
                nativeActorGrads.Dispose();
                nativeCriticGrads.Dispose();
            }
            
        }
    }

    void Dispose() {
        states.Dispose();
        actions.Dispose();
        log_probs.Dispose();
        returns.Dispose();
        advantages.Dispose();
        shuffle.Dispose();
        minibatches.Dispose();
        for (int i = 0; i < NUM_ENV; i++) {
            envs[i].Dispose();
        }
    }

    void Reset() {

    }
}

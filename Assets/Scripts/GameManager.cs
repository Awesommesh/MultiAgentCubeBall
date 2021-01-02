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
    public int BATCH_SIZE;
    public int MINI_BATCH_SIZE = 16;
    public int ADAM_BATCH_SIZE = 32;
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
    public NativeArray<int>[] layerShapes;
    public double[] actorLog_Std;
    public double[] criticLog_Std;
    public int episode_iteration = 0;
    public GameObject gameEnv;

    //PPO Experience Accumulation
    private NativeArray<double>[] states;
    private NativeArray<double>[] actions;
    private NativeArray<double>[] log_probs;
    private NativeArray<double> returns;
    private NativeArray<double> advantages;

    private NativeArray<int> minibatches;

    public Experience[] envs;

    //Statistic Tracking
    double actor_loss = 0;
    double critic_loss = 0;
    double total_loss = 0;
    double total_entropy = 0;
    void Awake() {
        Physics.autoSimulation = false;
        layerShapes = new NativeArray<int>[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layerShapes[i] = new NativeArray<int>(2, Allocator.Persistent);
        }
        //Have to hard code layer shapes ...
        //layer 1
        layerShapes[0][0] = 228;
        layerShapes[0][1] = STATE_SIZE;
        //layer 2
        layerShapes[1][0] = 684;
        layerShapes[1][1] = 228;
        //layer 3
        layerShapes[2][0] = 342;
        layerShapes[2][1] = 684;
        //layer 4
        layerShapes[3][0] = 171;
        layerShapes[3][1] = 342;
        //layer 5
        layerShapes[4][0] = 3;
        layerShapes[4][1] = 171;

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
                Vector3 ballOriginalPos = newBall.transform.position;
                Vector3[] blueOriginalPos = new Vector3[TEAM_SIZE];
                Vector3[] redOriginalPos = new Vector3[TEAM_SIZE];

                for (int k = 0; k < TEAM_SIZE; k++) {
                    newBlueTeam[k] = newEnv.transform.Find("BlueTeam/Blue"+(k+1)).gameObject;
                    blueOriginalPos[k] = newBlueTeam[k].transform.position;
                    newRedTeam[k] = newEnv.transform.Find("RedTeam/Red"+(k+1)).gameObject;
                    redOriginalPos[k] = newRedTeam[k].transform.position;
                }
                envs[i] = new Experience(newBall, newBlueGoal, newRedGoal, newBlueTeam, newRedTeam, 
                    ballOriginalPos, blueOriginalPos, redOriginalPos);
                numEnvCreated++;
            }
        }

        BATCH_SIZE = EPISODE_LENGTH * NUM_ENV * TEAM_SIZE * 2;
        minibatches = new NativeArray<int>(BATCH_SIZE, Allocator.Persistent);
        
        states = new NativeArray<double>[BATCH_SIZE];
        actions = new NativeArray<double>[BATCH_SIZE];
        log_probs = new NativeArray<double>[BATCH_SIZE];
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
            NativeArray<double>[] weights = new NativeArray<double>[numLayers];
            NativeArray<int>[] curShape = new NativeArray<int>[numLayers];
            for (int j = 0; j < numLayers; j++) {
                curShape[j] = new NativeArray<int>(numLayers, Allocator.Persistent);
                curShape[j][0] = layerShapes[j][0];
                curShape[j][1] = layerShapes[j][1];
                weights[j] = NativeNDOps.HeInitializedNDArray(layerShapes[j], layerShapes[j][1], Allocator.Persistent);
            }
            agents[i] = new NeuralNetwork(numLayers, activationList, ref weights, ref curShape, STATE_SIZE, NUM_ACTIONS, actorLog_Std, ALPHA, BETA1, BETA2, EPSILON, ADAM_BATCH_SIZE, BATCH_SIZE);
            weights = new NativeArray<double>[numLayers];
            curShape = new NativeArray<int>[numLayers];
            for (int j = 0; j < numLayers; j++) {
                curShape[j] = new NativeArray<int>(numLayers, Allocator.Persistent);
                if (j == numLayers - 1) {
                    curShape[j][0] = 1;
                } else {
                    curShape[j][0] = layerShapes[j][0];
                }
                curShape[j][1] = layerShapes[j][1];
                weights[j] = NativeNDOps.HeInitializedNDArray(layerShapes[j], layerShapes[j][1], Allocator.Persistent);
            }
            critics[i] = new NeuralNetwork(numLayers, activationList, ref weights, ref curShape, STATE_SIZE, 1, criticLog_Std, ALPHA, BETA1, BETA2, EPSILON, ADAM_BATCH_SIZE, BATCH_SIZE);
        }
        for (int i = 0; i < numLayers; i++) {
            layerShapes[i].Dispose();
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
                            states[globalInd] = envs[i].blueStates[k];
                            states[nextGlobalInd] = envs[i].redStates[k];
                            actions[globalInd] = envs[i].blueActions[k, j];
                            actions[nextGlobalInd] = envs[i].redActions[k, j];
                            log_probs[globalInd] = envs[i].blueLog_Probs[k, j];
                            log_probs[nextGlobalInd] = envs[i].redLog_Probs[k, j];
                            returns[globalInd] = envs[i].blueReturns[k*TEAM_SIZE + j];
                            advantages[globalInd] = envs[i].blueAdvantages[k*TEAM_SIZE + j];
                            returns[nextGlobalInd] = envs[i].redReturns[k*TEAM_SIZE + j];
                            advantages[nextGlobalInd] = envs[i].redAdvantages[k*TEAM_SIZE + j];
                            globalInd+=2;
                        }
                    }
                }

                actor_loss = 0;
                critic_loss = 0;
                total_loss = 0;
                int counter = 0;
                for (int i = 0; i < PPO_EPOCHS; i++) {
                    PPO_Update();
                    counter++;
                    Debug.Log("1 PPO Update");
                }
                actor_loss /= (BATCH_SIZE*NUM_ACTIONS*counter);
                actor_loss *= -1;
                critic_loss /= BATCH_SIZE*counter;
                total_entropy /= BATCH_SIZE*counter;
                total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * total_entropy;
                Debug.Log("Actor Loss: " + actor_loss + "\nCritic Loss: " + critic_loss + "\nTotal Loss: " + total_loss);
                
                for (int i = 0; i < NUM_ENV; i++) {
                    //Reset environment for next run;
                    envs[i].resetEnv();
                }
                //Need to reset agents adam optimizers?;
                /*for (int i = 0; i < TEAM_SIZE; i++) {
                    agents[i].resetOptimizerWeights();
                    critics[i].resetOptimizerWeights();
                }*/
                episode_iteration = 0;
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
            BATCH_SIZE = BATCH_SIZE
        };
        JobHandle jobHandle = batchJob.Schedule(numMiniBatches, 4);
        jobHandle.Complete();

        //PPO
        //Perform forward on each agent for each transition in batch
        NativeArray<double>[] actionDists = new NativeArray<double>[BATCH_SIZE];
        NativeArray<double>[] stateVals = new NativeArray<double>[BATCH_SIZE];
        for (int i = 0; i < TEAM_SIZE; i++) {
            //Perform forward network on every single transition
            NativeList<JobHandle> forwardJobHandles = new NativeList<JobHandle>(BATCH_SIZE * 2, Allocator.Persistent);
            for (int j = 0; j < BATCH_SIZE; j++) {
                actionDists[j] = new NativeArray<double>(NUM_ACTIONS, Allocator.Persistent);
                stateVals[j] = new NativeArray<double>(1, Allocator.Persistent);
                forwardJobHandles.Add(agents[i].Forward(states[j], ref actionDists[j], j));
                forwardJobHandles.Add(critics[i].Forward(states[j], ref stateVals[j], j));
            }
            JobHandle.CompleteAll(forwardJobHandles);
            forwardJobHandles.Dispose();
            
            for (int j = 0; j < numMiniBatches; j++) {
                for (int k = 0; k < MINI_BATCH_SIZE; k++) {
                    NativeList<JobHandle> backwardAndPPOJobHandles = new NativeList<JobHandle>(BATCH_SIZE * 3, Allocator.Persistent);
                    int index = minibatches[j*MINI_BATCH_SIZE+k];
                    NativeArray<double> actorGrads = new NativeArray<double>(NUM_ACTIONS, Allocator.Persistent);
                    NativeArray<double> criticGrads = new NativeArray<double>(1, Allocator.Persistent);
                    PPOUpdateJob ppoJob = new PPOUpdateJob {
                        actionDists = actionDists[index],
                        stateVal = stateVals[index][0],
                        actions = actions[index],
                        old_log_probs = log_probs[index],
                        returns = returns[index],
                        advantage = advantages[index],
                        log_stds= agents[i].log_std,
                        entropy = agents[i].entropy,
                        NUM_ACTIONS = NUM_ACTIONS,
                        PPO_EPILSON = PPO_EPILSON,
                        CRITIC_DISCOUNT = CRITIC_DISCOUNT,
                        ENTROPY_BETA = ENTROPY_BETA,
                        MINI_BATCH_SIZE = MINI_BATCH_SIZE,
                        actorGrad = actorGrads,
                        criticGrad = criticGrads[0],
                        actor_loss = actor_loss,
                        critic_loss = critic_loss
                    };
                    total_entropy += agents[i].entropy;
                    JobHandle ppoHandle = ppoJob.Schedule();
                    backwardAndPPOJobHandles.Add(ppoHandle);
                    backwardAndPPOJobHandles.Add(agents[i].Backward(actorGrads, index, ref ppoHandle));
                    backwardAndPPOJobHandles.Add(critics[i].Backward(criticGrads, index, ref ppoHandle));
                    JobHandle.CompleteAll(backwardAndPPOJobHandles);
                    backwardAndPPOJobHandles.Dispose();
                    actorGrads.Dispose();
                    criticGrads.Dispose();
                    agents[i].resetGrads();
                    critics[i].resetGrads();
                }
            }
        }
        for (int i = 0; i < BATCH_SIZE; i++) {
            actionDists[i].Dispose();
            stateVals[i].Dispose();
        }
    }

    void Dispose() {
        for (int i = 0; i < TEAM_SIZE; i++) {
            agents[i].Dispose();
            critics[i].Dispose();
        }
        //states.Dispose();
        //actions.Dispose();
        //log_probs.Dispose();
        returns.Dispose();
        advantages.Dispose();
        minibatches.Dispose();
        for (int i = 0; i < NUM_ENV; i++) {
            envs[i].Dispose();
        }
    }

    void Reset() {

    }
}

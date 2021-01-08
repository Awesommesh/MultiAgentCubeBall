using Unity.Mathematics;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using System;
public class GameManager : MonoBehaviour
{
    public static uint SEED = 17;
    public static uint NDArrayGenSeed = 2;
    public uint MINI_BATCH_SEED = 4;
    public int NUM_ENV;
    public static double GAE_LAMBDA = 0.95;
    public static double GAMMA = 0.99;
    public static double ALPHA = 0.0005;
    public static double BETA1 = 0.9;
    public static double BETA2 = 0.999;
    public static double EPSILON = 0.0000000001;
    public static int EPISODE_LENGTH = 128;
    public static int ITERATION = 1;
    public int BATCH_SIZE;
    public static int MINI_BATCH_SIZE = 256;
    public int NUM_MINI_BATCHES;
    public static int ADAM_BATCH_SIZE = 32;
    public int NUM_PPO_EPOCHS = 10;
    public static int TEAM_SIZE = 5;
    public static double MAX_SPEED = 25;
    public int numLayers;
    public static int STATE_SIZE = 76;
    public const double PPO_EPILSON = 0.2;
    public const double CRITIC_DISCOUNT = 0.5;
    public const double ENTROPY_BETA = 0.001;
    public int interval = 4;
    public static int NUM_ACTIONS = 3;
    public static float PHYSICS_STEP_SIZE = 0.01f;
    public const float X_Env_Increment = 74f;
    public const float Z_Env_Increment = 35f;
    public NeuralNetwork[] actors;
    public NeuralNetwork[] critics;
    public NativeArray<int>[] layerShapes;
    public double[] actorStd;
    public double[] criticStd;
    public int episode_iteration = 0;
    public GameObject gameEnv;

    //PPO Experience Accumulation
    private NativeArray<double> batchStates;
    private NativeArray<double> batchActions;
    private NativeArray<double> batchLog_Probs;
    private NativeArray<double> batchReturns;
    private NativeArray<double> batchAdvantages;
    private NativeArray<double>[] states;
    private NativeArray<double>[] actions;
    private NativeArray<double>[] log_probs;
    private NativeArray<double>[] returns;
    private NativeArray<double>[] advantages;

    //PPO Update Intermediates
    NativeArray<double>[] actionDists = new NativeArray<double>[TEAM_SIZE];
    NativeArray<double>[] stateVals = new NativeArray<double>[TEAM_SIZE];
    NativeArray<double>[] actorGrads = new NativeArray<double>[TEAM_SIZE];
    NativeArray<double>[] criticGrads = new NativeArray<double>[TEAM_SIZE];

    private NativeArray<int>[] minibatches;

    public Experience[] envs;

    //Statistic Tracking
    double actor_loss = 0;
    double critic_loss = 0;
    double total_loss = 0;
    double total_entropy = 0;

    Unity.Mathematics.Random randSeed;

    [BurstCompile]
    void Awake() {
        Physics.autoSimulation = false;
        ITERATION = 1;
        BATCH_SIZE = EPISODE_LENGTH * NUM_ENV * TEAM_SIZE * 2;
        NUM_MINI_BATCHES = BATCH_SIZE / MINI_BATCH_SIZE;
        minibatches = new NativeArray<int>[NUM_PPO_EPOCHS];
        int trueBatchSize = NUM_MINI_BATCHES*MINI_BATCH_SIZE;
        randSeed = new Unity.Mathematics.Random(MINI_BATCH_SEED);
        
        states = new NativeArray<double>[NUM_MINI_BATCHES];
        actions = new NativeArray<double>[NUM_MINI_BATCHES];
        log_probs = new NativeArray<double>[NUM_MINI_BATCHES];
        returns = new NativeArray<double>[NUM_MINI_BATCHES];
        advantages = new NativeArray<double>[NUM_MINI_BATCHES];
        batchStates = new NativeArray<double>(trueBatchSize*STATE_SIZE, Allocator.Persistent);
        batchActions= new NativeArray<double>(trueBatchSize*NUM_ACTIONS, Allocator.Persistent);
        batchLog_Probs = new NativeArray<double>(trueBatchSize*NUM_ACTIONS, Allocator.Persistent);
        batchReturns = new NativeArray<double>(trueBatchSize, Allocator.Persistent);
        batchAdvantages = new NativeArray<double>(trueBatchSize, Allocator.Persistent);

        for (int i = 0; i < NUM_MINI_BATCHES; i++) {
            states[i] = new NativeArray<double>(MINI_BATCH_SIZE*STATE_SIZE, Allocator.Persistent);
            actions[i] = new NativeArray<double>(MINI_BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
            log_probs[i] = new NativeArray<double>(MINI_BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
            returns[i] = new NativeArray<double>(MINI_BATCH_SIZE, Allocator.Persistent);
            advantages[i] = new NativeArray<double>(MINI_BATCH_SIZE, Allocator.Persistent);
        }

        for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
            minibatches[i] = new NativeArray<int>(trueBatchSize, Allocator.Persistent);
        }

        for (int i = 0; i < TEAM_SIZE; i++) {
            actionDists[i] = new NativeArray<double>(MINI_BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
            stateVals[i] = new NativeArray<double>(MINI_BATCH_SIZE, Allocator.Persistent);
            actorGrads[i] = new NativeArray<double>(MINI_BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
            criticGrads[i] = new NativeArray<double>(MINI_BATCH_SIZE, Allocator.Persistent);
        }

        episode_iteration = 0;

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
                GameObject newBall = newEnv.transform.Find("Soccer Ball").gameObject;
                GameObject newBlueGoal = newEnv.transform.Find("GoalBlue").gameObject;
                GameObject newRedGoal = newEnv.transform.Find("GoalRed").gameObject;
                GameObject[] newBlueTeam = new GameObject[TEAM_SIZE];
                GameObject[] newRedTeam = new GameObject[TEAM_SIZE];
                Vector3 ballOriginalPos = newBall.transform.position;
                Vector3[] blueOriginalPos = new Vector3[TEAM_SIZE];
                Vector3[] redOriginalPos = new Vector3[TEAM_SIZE];

                for (int k = 0; k < TEAM_SIZE; k++) {
                    newBlueTeam[k] = newEnv.transform.Find("Blue"+(k+1)).gameObject;
                    blueOriginalPos[k] = newBlueTeam[k].transform.position;
                    newRedTeam[k] = newEnv.transform.Find("Red"+(k+1)).gameObject;
                    redOriginalPos[k] = newRedTeam[k].transform.position;
                }
                envs[numEnvCreated] = new Experience(newBall, newBlueGoal, newRedGoal, newBlueTeam, newRedTeam, 
                    ballOriginalPos, blueOriginalPos, redOriginalPos, (uint)((numEnvCreated+1)*117));
                numEnvCreated++;
            }
        }
        
        //Initialize NN randomly with correct architecture etc.
        layerShapes = new NativeArray<int>[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layerShapes[i] = new NativeArray<int>(2, Allocator.Persistent);
        }
        //Have to hard code layer shapes ...
        //layer 1
        layerShapes[0][0] = 228;
        layerShapes[0][1] = STATE_SIZE+1;
        //layer 2
        layerShapes[1][0] = 684;
        layerShapes[1][1] = 228+1;
        //layer 3
        layerShapes[2][0] = 342;
        layerShapes[2][1] = 684+1;
        //layer 4
        layerShapes[3][0] = 171;
        layerShapes[3][1] = 342+1;
        //layer 5
        layerShapes[4][0] = 3;
        layerShapes[4][1] = 171+1;

        actorStd = new double[NUM_ACTIONS];
        for (int i = 0; i < NUM_ACTIONS; i++) {
            actorStd[i] = 10;
        }
        criticStd = new double[1];
        criticStd[0] = 1;
        actors = new NeuralNetwork[TEAM_SIZE];
        critics = new NeuralNetwork[TEAM_SIZE];

        ActivationType[] actorActivationList = new ActivationType[numLayers];
        actorActivationList[0] = ActivationType.ReLU;
        actorActivationList[1] = ActivationType.ReLU;
        actorActivationList[2] = ActivationType.ReLU;
        actorActivationList[3] = ActivationType.ReLU;
        actorActivationList[4] = ActivationType.None;
        ActivationType[] criticActivationList = new ActivationType[numLayers];
        criticActivationList[0] = ActivationType.ReLU;
        criticActivationList[1] = ActivationType.ReLU;
        criticActivationList[2] = ActivationType.ReLU;
        criticActivationList[3] = ActivationType.ReLU;
        criticActivationList[4] = ActivationType.None;
        
        for (int i = 0; i < TEAM_SIZE; i++) {
            NativeArray<double>[] weights = new NativeArray<double>[numLayers];
            NativeArray<int>[] curShape = new NativeArray<int>[numLayers];
            for (int j = 0; j < numLayers; j++) {
                curShape[j] = new NativeArray<int>(numLayers, Allocator.Persistent);
                curShape[j][0] = layerShapes[j][0];
                curShape[j][1] = layerShapes[j][1];
                weights[j] = NativeNDOps.HeInitializedNDArray(layerShapes[j], layerShapes[j][1], Allocator.Persistent);
            }
            actors[i] = new NeuralNetwork(numLayers, actorActivationList, weights, curShape, STATE_SIZE, NUM_ACTIONS, 
                actorStd, ALPHA, BETA1, BETA2, EPSILON, ADAM_BATCH_SIZE, 2, MINI_BATCH_SIZE);
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
            
            critics[i] = new NeuralNetwork(numLayers, criticActivationList, weights, curShape, STATE_SIZE, 1, 
                criticStd, ALPHA, BETA1, BETA2, EPSILON, ADAM_BATCH_SIZE, 2, MINI_BATCH_SIZE);
        }
        for (int i = 0; i < numLayers; i++) {
            layerShapes[i].Dispose();
        }
    }

    [BurstCompile]
    // Update is called once per frame
    void Update()
    {
        if (Time.frameCount % interval == 0) {
            if (episode_iteration < EPISODE_LENGTH) {
                for (int i = 0; i < NUM_ENV; i++) {
                    envs[i].stepForward(actors, critics, actors, critics);
                }
                episode_iteration++;
            } else if (episode_iteration == EPISODE_LENGTH) { //Learn from experiences...
                Debug.Log("Began Training...");
                actor_loss = 0;
                critic_loss = 0;
                total_loss = 0;

                //read from all the environments
                int curInd = 0;
                for (int e = 0; e < NUM_ENV; e++) {
                    //Get final values and calc gae
                    envs[e].getNextValues(critics, critics);
                    envs[e].CalculateGAE();
                    //Merge current environment data to form first set of minibatches
                    //For each player on given env append all of their experiences in random order
                    for (int j = 0; j < TEAM_SIZE; j++) {
                        for (int k = 0; k < EPISODE_LENGTH; k++) {
                            if (curInd >= NUM_MINI_BATCHES * MINI_BATCH_SIZE) {
                                Debug.Log("NOTE: HAD TO SKIP SOME EXPERIENCE DATA BECAUSE BATCH_SIZE WASN'T DIVISIBLE BY MINI_BATCH_SIZE. Missed " 
                                    + (BATCH_SIZE-NUM_MINI_BATCHES * MINI_BATCH_SIZE) + " transitions...");
                                break;
                            }
                            int nextInd = curInd + 1;
                            //Blue and Red Agent j's kth time_step
                            for (int l = 0; l < STATE_SIZE; l++) {
                                batchStates[curInd*STATE_SIZE + l] = envs[e].blueStates[k][l];
                                batchStates[nextInd*STATE_SIZE + l] = envs[e].redStates[k][l];
                            }
                            for (int l = 0; l < NUM_ACTIONS; l++) {
                                batchActions[curInd*NUM_ACTIONS + l] = envs[e].blueActions[k, j][l];
                                batchActions[nextInd*NUM_ACTIONS + l] = envs[e].redActions[k, j][l];
                                batchLog_Probs[curInd*NUM_ACTIONS + l] = envs[e].blueLog_Probs[k, j][l];
                                batchLog_Probs[nextInd*NUM_ACTIONS + l] = envs[e].redLog_Probs[k, j][l];
                            }
                            batchReturns[curInd] = envs[e].blueReturns[k*TEAM_SIZE + j];
                            batchReturns[nextInd] = envs[e].redReturns[k*TEAM_SIZE + j];
                            batchAdvantages[curInd] = envs[e].blueAdvantages[k*TEAM_SIZE + j];
                            batchAdvantages[nextInd] = envs[e].redAdvantages[k*TEAM_SIZE + j];
                            curInd+=2;
                        }
                    }
                }
                //Generate set of minibatches randomly for each PPO EPOCH
                NativeArray<JobHandle> generateMiniBatchesJobHandles = new NativeArray<JobHandle>(NUM_PPO_EPOCHS, Allocator.Persistent);
                for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
                    //Generate set of minibatches randomly
                    GenerateMiniBatchesJob batchJob = new GenerateMiniBatchesJob {
                        seed = (uint)(randSeed.NextDouble()*100000+11),
                        MINI_BATCH_SIZE = MINI_BATCH_SIZE,
                        minibatches = minibatches[i],
                        BATCH_SIZE = NUM_MINI_BATCHES*MINI_BATCH_SIZE
                    };
                    generateMiniBatchesJobHandles[i] = batchJob.Schedule(NUM_MINI_BATCHES, 4);
                }
                JobHandle.CompleteAll(generateMiniBatchesJobHandles);
                generateMiniBatchesJobHandles.Dispose();

                for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
                    for (int j = 0; j < NUM_MINI_BATCHES; j++) {
                        for (int k = 0; k < MINI_BATCH_SIZE; k++) {
                            int index = minibatches[i][j*MINI_BATCH_SIZE + k];
                            for (int l = 0; l < STATE_SIZE; l++) {
                                states[j][l*MINI_BATCH_SIZE + k] = batchStates[index*STATE_SIZE+l];
                            }
                            for (int l = 0; l < NUM_ACTIONS; l++) {
                                actions[j][l*MINI_BATCH_SIZE + k] = batchActions[index*NUM_ACTIONS+l];
                                log_probs[j][l*MINI_BATCH_SIZE + k] = batchLog_Probs[index*NUM_ACTIONS+l];
                            }
                            returns[j][k] = batchReturns[index];
                            advantages[j][k] = batchAdvantages[index];
                        }
                    }
                    PPO_Update(i);
                }
                for (int i = 0; i < TEAM_SIZE; i++) {
                    actors[i].resetOptimizerWeights();
                    critics[i].resetOptimizerWeights();
                }
                double blueTeamsAvg = 0;
                double redTeamsAvg = 0;
                for (int i = 0; i < NUM_ENV; i++) {
                    blueTeamsAvg += envs[i].avgBlueReward;
                    redTeamsAvg += envs[i].avgRedReward;
                    //Debug.Log("Environment " + i + " blue team average rewards: " + envs[i].avgBlueReward);
                    //Debug.Log("Environment " + i + " red team average rewards: " + envs[i].avgRedReward);
                }
                blueTeamsAvg /= NUM_ENV;
                redTeamsAvg /= NUM_ENV;
                Debug.Log("Blue Teams' Average Rewards: " + blueTeamsAvg);
                Debug.Log("Red Teams' Average Rewards: " + redTeamsAvg);
                actor_loss /= (MINI_BATCH_SIZE*NUM_MINI_BATCHES*NUM_ACTIONS*TEAM_SIZE*NUM_PPO_EPOCHS);
                critic_loss /= (MINI_BATCH_SIZE*NUM_MINI_BATCHES*TEAM_SIZE*NUM_PPO_EPOCHS);
                total_entropy /= (MINI_BATCH_SIZE*NUM_MINI_BATCHES*TEAM_SIZE*NUM_PPO_EPOCHS);
                total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * total_entropy;
                Debug.Log("Actor Loss: " + actor_loss 
                    + "\nCritic Loss: " + critic_loss
                    + "\nTotal Entropy: " + total_entropy);
                Debug.Log("Total Loss: " + total_loss);                
                for (int i = 0; i < NUM_ENV; i++) {
                    //Reset environment for next run;
                    envs[i].resetEnv();
                }
                episode_iteration = 0;
                ITERATION++;
            }
            Physics.Simulate(PHYSICS_STEP_SIZE);
        }
    }

    [BurstCompile]
    void PPO_Update(int epoch) {
        //Debug.Log("Starting PPO EPOCH");
        for (int i = 0; i < NUM_MINI_BATCHES; i++) {
            NativeList<JobHandle> forwardJobHandles = new NativeList<JobHandle>(TEAM_SIZE*2, Allocator.Persistent);
            //if (i == 0) {
                /*for (int j = 0; j < states[i].Length; j++) {
                    Debug.Log(states[i][j]);
                }*/
            //}
            for (int j = 0; j < TEAM_SIZE; j++) {
                forwardJobHandles.Add(actors[j].Forward(states[i], MINI_BATCH_SIZE, actionDists[j], 0));
                forwardJobHandles.Add(critics[j].Forward(states[i], MINI_BATCH_SIZE, stateVals[j], 0));
            }
            JobHandle.CompleteAll(forwardJobHandles);
            forwardJobHandles.Dispose();
            NativeList<JobHandle> backwardAndPPOJobHandles = new NativeList<JobHandle>(TEAM_SIZE*3, Allocator.Persistent);
            NativeArray<double>[] AL = new NativeArray<double>[TEAM_SIZE];
            NativeArray<double>[] CL = new NativeArray<double>[TEAM_SIZE];
            //NativeArray<double>[] entropyL = new NativeArray<double>[TEAM_SIZE];
            for (int j = 0; j < TEAM_SIZE; j++) {
                total_entropy += actors[j].entropy*MINI_BATCH_SIZE;
                AL[j] = new NativeArray<double>(1, Allocator.Persistent);
                CL[j] = new NativeArray<double>(1, Allocator.Persistent);
                if (i == 0 && epoch == 1) {
                    /*for (int k = 0; k < actionDists[j].Length; k++) {
                        if (math.abs(actionDists[j][k]) > 100) {
                            Debug.Log(j + "wtf"+(k/MINI_BATCH_SIZE));
                        }
                    }*/
                    /*for (int k = 0; k < stateVals[j].Length; k++) {
                        Debug.Log(math.log(stateVals[j][k]/(1-stateVals[j][k])));
                    }*/
                }
                //entropyL = new NativeArray<double>(1, Allocator.Persistent);
                PPOUpdateJob ppoJob = new PPOUpdateJob {
                    actionDists = actionDists[j],
                    stateVal = stateVals[j],
                    actions = actions[i],
                    old_log_probs = log_probs[i],
                    returns = returns[i],
                    advantage = advantages[i],
                    stds = actors[j].std,
                    entropy = actors[j].entropy,
                    NUM_ACTIONS = NUM_ACTIONS,
                    PPO_EPILSON = PPO_EPILSON,
                    CRITIC_DISCOUNT = CRITIC_DISCOUNT,
                    ENTROPY_BETA = ENTROPY_BETA,
                    MINI_BATCH_SIZE = MINI_BATCH_SIZE,
                    actorGrad = actorGrads[j],
                    criticGrad = criticGrads[j],
                    actor_loss = AL[j],
                    critic_loss = CL[j],
                };
                JobHandle ppoHandle = ppoJob.Schedule();
                backwardAndPPOJobHandles.Add(ppoHandle);
                backwardAndPPOJobHandles.Add(actors[j].Backward(actorGrads[j], MINI_BATCH_SIZE, 0, ppoHandle));
                backwardAndPPOJobHandles.Add(critics[j].Backward(criticGrads[j], MINI_BATCH_SIZE, 0, ppoHandle));
            }
            JobHandle.CompleteAll(backwardAndPPOJobHandles);
            backwardAndPPOJobHandles.Dispose();
            for (int j = 0; j < TEAM_SIZE; j++) {
                actor_loss += AL[j][0];
                critic_loss += CL[j][0];
                //total_entropy += entropyL[j][0];
                AL[j].Dispose();
                CL[j].Dispose();
                //entropyL[j].Dispose();
                actors[j].resetGrads();
                critics[j].resetGrads();
            }
        }        
    }

    void OnApplicationQuit()
    {
        Dispose();
    }

    [BurstCompile]
    void Dispose() {
        for (int i = 0; i < TEAM_SIZE; i++) {
            actors[i].Dispose();
            critics[i].Dispose();
        }
        batchStates.Dispose();
        batchActions.Dispose();
        batchLog_Probs.Dispose();
        batchReturns.Dispose();
        batchAdvantages.Dispose();
        for (int i = 0; i < NUM_MINI_BATCHES; i++) {
            states[i].Dispose();
            actions[i].Dispose();
            log_probs[i].Dispose();
            returns[i].Dispose();
            advantages[i].Dispose();
        }
        for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
            minibatches[i].Dispose();
        }

        for (int i = 0; i < TEAM_SIZE; i++) {
            actionDists[i].Dispose();
            stateVals[i].Dispose();
            actorGrads[i].Dispose();
            criticGrads[i].Dispose();
        }

        for (int i = 0; i < NUM_ENV; i++) {
            envs[i].Dispose();
        }
        Debug.Log("Disposed everything...");
    }

    void Reset() {

    }
}

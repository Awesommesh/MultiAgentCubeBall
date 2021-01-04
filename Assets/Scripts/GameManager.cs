using Unity.Mathematics;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using System;
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
    public static int EPISODE_LENGTH = 128;
    public int BATCH_SIZE;
    public int MINI_BATCH_SIZE = 16;
    public int NUM_MINI_BATCHES;
    public int ADAM_BATCH_SIZE = 32;
    public int NUM_PPO_EPOCHS = 10;
    public static int TEAM_SIZE = 5;
    public static double FIELD_LENGTH = 90;
    public static double MAX_SPEED = 25;
    public int numLayers;
    public static int STATE_SIZE = 76;
    public const double PPO_EPILSON = 0.2;
    public const double CRITIC_DISCOUNT = 0.5;
    public const double ENTROPY_BETA = 0.001;
    public int interval = 4;
    public static int NUM_ACTIONS = 3;
    public static float PHYSICS_STEP_SIZE = 0.01f;
    public const float X_Env_Increment = 36f;
    public const float Z_Env_Increment = 18f;
    public NeuralNetwork[] agents;
    public NeuralNetwork[] critics;
    public NativeArray<int>[] layerShapes;
    public double[] actorLog_Std;
    public double[] criticLog_Std;
    public int episode_iteration = 0;
    public GameObject gameEnv;

    //PPO Experience Accumulation
    private NativeArray<double>[,] states;
    private NativeArray<double>[,] actions;
    private NativeArray<double>[,] log_probs;
    private NativeArray<double>[,] returns;
    private NativeArray<double>[,] advantages;

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

    [BurstCompile]
    void Awake() {
        Physics.autoSimulation = false;
        BATCH_SIZE = EPISODE_LENGTH * NUM_ENV * TEAM_SIZE * 2;
        NUM_MINI_BATCHES = BATCH_SIZE / MINI_BATCH_SIZE;
        minibatches = new NativeArray<int>[NUM_PPO_EPOCHS];
        
        states = new NativeArray<double>[NUM_PPO_EPOCHS, NUM_MINI_BATCHES];
        actions = new NativeArray<double>[NUM_PPO_EPOCHS, NUM_MINI_BATCHES];
        log_probs = new NativeArray<double>[NUM_PPO_EPOCHS, NUM_MINI_BATCHES];
        returns = new NativeArray<double>[NUM_PPO_EPOCHS, NUM_MINI_BATCHES];
        advantages = new NativeArray<double>[NUM_PPO_EPOCHS, NUM_MINI_BATCHES];
        for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
            for (int j = 0; j < NUM_MINI_BATCHES; j++) {
                states[i, j] = new NativeArray<double>(MINI_BATCH_SIZE*STATE_SIZE, Allocator.Persistent);
                actions[i, j] = new NativeArray<double>(MINI_BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
                log_probs[i, j] = new NativeArray<double>(MINI_BATCH_SIZE*NUM_ACTIONS, Allocator.Persistent);
                returns[i, j] = new NativeArray<double>(MINI_BATCH_SIZE, Allocator.Persistent);
                advantages[i, j] = new NativeArray<double>(MINI_BATCH_SIZE, Allocator.Persistent);
            }
            minibatches[i] = new NativeArray<int>(NUM_MINI_BATCHES*MINI_BATCH_SIZE, Allocator.Persistent);
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
                GameObject newBlueGoal = newEnv.transform.Find("Field/GoalAreaBlue").gameObject;
                GameObject newRedGoal = newEnv.transform.Find("Field/GoalAreaRed").gameObject;
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
                envs[numEnvCreated] = new Experience(newBall, newBlueGoal, newRedGoal, newBlueTeam, newRedTeam, 
                    ballOriginalPos, blueOriginalPos, redOriginalPos);
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
            actorLog_Std[i] = 1;
        }
        criticLog_Std = new double[1];
        criticLog_Std[0] = 1;
        agents = new NeuralNetwork[TEAM_SIZE];
        critics = new NeuralNetwork[TEAM_SIZE];

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
            agents[i] = new NeuralNetwork(numLayers, activationList, ref weights, ref curShape, STATE_SIZE, NUM_ACTIONS, 
                actorLog_Std, ALPHA, BETA1, BETA2, EPSILON, ADAM_BATCH_SIZE, BATCH_SIZE, MINI_BATCH_SIZE);
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
            critics[i] = new NeuralNetwork(numLayers, activationList, ref weights, ref curShape, STATE_SIZE, 1, 
                criticLog_Std, ALPHA, BETA1, BETA2, EPSILON, ADAM_BATCH_SIZE, BATCH_SIZE, MINI_BATCH_SIZE);
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
                    envs[i].stepForward(agents, critics, agents, critics);
                }
                episode_iteration++;
            } else if (episode_iteration == EPISODE_LENGTH) { //Learn from experiences...
                Debug.Log("Begun Training...");
                actor_loss = 0;
                critic_loss = 0;
                total_loss = 0;

                //Generate set of minibatches randomly for each PPO EPOCH
                NativeArray<JobHandle> generateMiniBatchesJobHandles = new NativeArray<JobHandle>(NUM_PPO_EPOCHS, Allocator.Persistent);
                for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
                    //Generate set of minibatches randomly
                    GenerateMiniBatchesJob batchJob = new GenerateMiniBatchesJob {
                        seed = MINI_BATCH_SEED,
                        MINI_BATCH_SIZE = MINI_BATCH_SIZE,
                        minibatches = minibatches[i],
                        BATCH_SIZE = BATCH_SIZE
                    };
                    generateMiniBatchesJobHandles[i] = batchJob.Schedule(NUM_MINI_BATCHES, 4);
                }
                JobHandle.CompleteAll(generateMiniBatchesJobHandles);
                generateMiniBatchesJobHandles.Dispose();

                for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
                    //Setup Data in mini_batches based on generate mini batches output
                    if (i == 0) { 
                        //First epoch have to read from all the environments
                        for (int e = 0; e < NUM_ENV; e++) {
                            //Get final values and calc gae
                            envs[e].getNextValues(critics, critics);
                            envs[e].CalculateGAE();
                            //Merge current environment data to form first set of minibatches
                            //For each player on given env append all of their experiences in random order
                            int curInd = 0;
                            for (int j = 0; j < TEAM_SIZE; j++) {
                                for (int k = 0; k < EPISODE_LENGTH; k++) {
                                    if (curInd >= NUM_MINI_BATCHES * MINI_BATCH_SIZE) {
                                        Debug.Log("NOTE: HAD TO SKIP SOME EXPERIENCE DATA BECAUSE BATCH_SIZE WASN'T DIVISIBLE BY MINI_BATCH_SIZE. Missed " 
                                            + (BATCH_SIZE-NUM_MINI_BATCHES * MINI_BATCH_SIZE) + " transitions...");
                                        break;
                                    }
                                    int batchNo = curInd / MINI_BATCH_SIZE;
                                    int index = minibatches[i][curInd];
                                    int nextIndex = minibatches[i][curInd + 1];

                                    //Blue and Red Agent j's kth time_step
                                    for (int l = 0; l < STATE_SIZE; l++) {
                                        states[i, batchNo][l*MINI_BATCH_SIZE + (index % MINI_BATCH_SIZE)] = envs[e].blueStates[k][l];
                                        states[i, batchNo][l*MINI_BATCH_SIZE + (nextIndex % MINI_BATCH_SIZE)] = envs[e].redStates[k][l];
                                    }
                                    for (int l = 0; l < NUM_ACTIONS; l++) {
                                        actions[i, batchNo][l*MINI_BATCH_SIZE + (index % MINI_BATCH_SIZE)] = envs[e].blueActions[k, j][l];
                                        actions[i, batchNo][l*MINI_BATCH_SIZE + (nextIndex % MINI_BATCH_SIZE)] = envs[e].redActions[k, j][l];
                                        log_probs[i, batchNo][l*MINI_BATCH_SIZE + (index % MINI_BATCH_SIZE)] = envs[e].blueLog_Probs[k, j][l];
                                        log_probs[i, batchNo][l*MINI_BATCH_SIZE + (nextIndex % MINI_BATCH_SIZE)] = envs[e].redLog_Probs[k, j][l];
                                    }
                                    
                                    returns[i, batchNo][(index % MINI_BATCH_SIZE)] = envs[e].blueReturns[k*TEAM_SIZE + j];
                                    advantages[i, batchNo][(index % MINI_BATCH_SIZE)] = envs[e].blueAdvantages[k*TEAM_SIZE + j];
                                    returns[i, batchNo][(nextIndex % MINI_BATCH_SIZE)] = envs[e].redReturns[k*TEAM_SIZE + j];
                                    advantages[i, batchNo][(nextIndex % MINI_BATCH_SIZE)] = envs[e].redAdvantages[k*TEAM_SIZE + j];
                                    curInd+=2;
                                }
                            }
                        }
                    } else {
                        //Select from previous epoch gathered data
                        for (int j = 0; j < NUM_MINI_BATCHES * MINI_BATCH_SIZE; j++) {
                            int index = minibatches[i][j];
                            int curBatchNo = j / MINI_BATCH_SIZE;
                            int prevBatchNo = index / MINI_BATCH_SIZE;
                            for (int k = 0; k < STATE_SIZE; k++) {
                                states[i, curBatchNo][k*MINI_BATCH_SIZE + (j % MINI_BATCH_SIZE)] = states[i-1, prevBatchNo][k*MINI_BATCH_SIZE + (index % MINI_BATCH_SIZE)];
                            }
                            for (int k = 0; k < NUM_ACTIONS; k++) {
                                actions[i, curBatchNo][k*MINI_BATCH_SIZE + (j % MINI_BATCH_SIZE)] = actions[i-1, prevBatchNo][k*MINI_BATCH_SIZE + (index % MINI_BATCH_SIZE)];
                                log_probs[i, curBatchNo][k*MINI_BATCH_SIZE + (j % MINI_BATCH_SIZE)] = log_probs[i-1, prevBatchNo][k*MINI_BATCH_SIZE + (index % MINI_BATCH_SIZE)];
                            }
                            returns[i, curBatchNo][j % MINI_BATCH_SIZE] = returns[i-1, prevBatchNo][index % MINI_BATCH_SIZE];
                            advantages[i, curBatchNo][j % MINI_BATCH_SIZE] = advantages[i-1, prevBatchNo][index % MINI_BATCH_SIZE];
                        }
                        //Dispose previous ppo_epoch data
                        //DisposePPOEpochData(i-1);
                    }
                    PPO_Update(i);
                }
                for (int i = 0; i < TEAM_SIZE; i++) {
                    agents[i].resetOptimizerWeights();
                    critics[i].resetOptimizerWeights();
                }
                actor_loss /= (MINI_BATCH_SIZE*NUM_MINI_BATCHES*NUM_ACTIONS*TEAM_SIZE*NUM_PPO_EPOCHS);
                critic_loss /= (MINI_BATCH_SIZE*NUM_MINI_BATCHES*TEAM_SIZE*NUM_PPO_EPOCHS);
                total_entropy /= (MINI_BATCH_SIZE*NUM_MINI_BATCHES*NUM_PPO_EPOCHS);
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

    [BurstCompile]
    void PPO_Update(int epoch) {
        //Debug.Log("Starting PPO EPOCH");
        for (int i = 0; i < NUM_MINI_BATCHES; i++) {
            NativeList<JobHandle> forwardJobHandles = new NativeList<JobHandle>(TEAM_SIZE*2, Allocator.Persistent);
            for (int j = 0; j < TEAM_SIZE; j++) {
                forwardJobHandles.Add(agents[j].Forward(states[epoch, i], MINI_BATCH_SIZE, ref actionDists[j], j));
                forwardJobHandles.Add(critics[j].Forward(states[epoch, i], MINI_BATCH_SIZE, ref stateVals[j], j));
            }
            JobHandle.CompleteAll(forwardJobHandles);
            forwardJobHandles.Dispose();

            NativeList<JobHandle> backwardAndPPOJobHandles = new NativeList<JobHandle>(TEAM_SIZE*3, Allocator.Persistent);
            NativeArray<double>[] AL = new NativeArray<double>[TEAM_SIZE];
            NativeArray<double>[] CL = new NativeArray<double>[TEAM_SIZE];
            for (int j = 0; j < TEAM_SIZE; j++) {
                total_entropy += agents[j].entropy*MINI_BATCH_SIZE;
                AL[j] = new NativeArray<double>(1, Allocator.Persistent);
                CL[j] = new NativeArray<double>(1, Allocator.Persistent);
                PPOUpdateJob ppoJob = new PPOUpdateJob {
                    actionDists = actionDists[j],
                    stateVal = stateVals[j],
                    actions = actions[epoch, i],
                    old_log_probs = log_probs[epoch, i],
                    returns = returns[epoch, i],
                    advantage = advantages[epoch, i],
                    log_stds= agents[j].log_std,
                    entropy = agents[j].entropy,
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
                backwardAndPPOJobHandles.Add(agents[j].Backward(actorGrads[j], MINI_BATCH_SIZE, j, ref ppoHandle));
                backwardAndPPOJobHandles.Add(critics[j].Backward(criticGrads[j], MINI_BATCH_SIZE, j, ref ppoHandle));
            }
            JobHandle.CompleteAll(backwardAndPPOJobHandles);
            backwardAndPPOJobHandles.Dispose();
            for (int j = 0; j < TEAM_SIZE; j++) {
                actor_loss += AL[j][0];
                critic_loss += CL[j][0];
                AL[j].Dispose();
                CL[j].Dispose();
                agents[j].resetGrads();
                critics[j].resetGrads();
            }
        }        
    }

    /*[BurstCompile]
    void DisposePPOEpochData(int epoch) {
        for (int i = 0; i < NUM_MINI_BATCHES; i++) {
            states[epoch, i].Dispose();
            actions[epoch, i].Dispose();
            log_probs[epoch, i].Dispose();
            returns[epoch, i].Dispose();
            advantages[epoch, i].Dispose();
        }
    }*/

    void OnApplicationQuit()
    {
        Dispose();
    }

    [BurstCompile]
    void Dispose() {
        for (int i = 0; i < TEAM_SIZE; i++) {
            agents[i].Dispose();
            critics[i].Dispose();
        }
        for (int i = 0; i < NUM_PPO_EPOCHS; i++) {
            for (int j = 0; j < NUM_MINI_BATCHES; j++) {
                states[i, j].Dispose();
                actions[i, j].Dispose();
                log_probs[i, j].Dispose();
                returns[i, j].Dispose();
                advantages[i, j].Dispose();
            }
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
    }

    void Reset() {

    }
}

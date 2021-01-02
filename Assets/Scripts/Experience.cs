using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using UnityEngine;
using Unity.Burst;
public class Experience {
    //Blue Team
    public NDArray blueRewards;
    public NativeArray<double>[,] blueValues;
    public NativeArray<double>[] blueStates;
    public NativeArray<double>[,] blueActions;
    public NativeArray<double>[,] blueLog_Probs;
    public NativeArray<double>[] blueNextVals;
    public NativeArray<double> blueReturns;
    public NativeArray<double> blueAdvantages;
    GameObject[] bluePlayers;
    private Vector3[] blueOriginalPos;
    public GameObject blueGoal;
    public bool blueWon;

    //Red Team   
    public NDArray redRewards;
    public NativeArray<double>[,] redValues;
    public NativeArray<double>[] redStates;
    public NativeArray<double>[,] redActions;
    public NativeArray<double>[,] redLog_Probs;
    public NativeArray<double>[] redNextVals;
    public NativeArray<double> redReturns;
    public NativeArray<double> redAdvantages;
    GameObject[] redPlayers;
    private Vector3[] redOriginalPos;
    public GameObject redGoal;
    public bool redWon;

    //General
    GameObject ball;
    Vector3 ballOriginalPos;
    public NativeArray<int> mask;
    int time_step = 0;
    
    public Experience (GameObject ball, GameObject blueGoal, GameObject redGoal, GameObject[] blueTeam, GameObject[] redTeam, 
        Vector3 ballOriginalPos, Vector3[] blueOriginalPos, Vector3[] redOriginalPos) {
        //General Initialization
        time_step = 0;
        this.ball = ball;
        this.ballOriginalPos = ballOriginalPos;
        mask = new NativeArray<int>(GameManager.EPISODE_LENGTH, Allocator.Persistent);

        //Blue initialization
        blueRewards = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE);
        blueValues = new NativeArray<double>[GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE];
        blueStates = new NativeArray<double>[GameManager.EPISODE_LENGTH]; 
        blueActions = new NativeArray<double>[GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE];
        blueLog_Probs = new NativeArray<double>[GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE];
        blueNextVals = new NativeArray<double>[GameManager.TEAM_SIZE];
        blueReturns = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        blueAdvantages = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        bluePlayers = blueTeam;
        this.blueOriginalPos = blueOriginalPos;
        this.blueGoal = blueGoal;
        blueWon = false;

        //Red Initialization
        redRewards = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE);
        redValues = new NativeArray<double>[GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE];
        redStates = new NativeArray<double>[GameManager.EPISODE_LENGTH];
        redActions = new NativeArray<double>[GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE];
        redLog_Probs = new NativeArray<double>[GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE];
        redNextVals = new NativeArray<double>[GameManager.TEAM_SIZE];
        redReturns = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        redAdvantages = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        redPlayers = redTeam;
        this.redOriginalPos = redOriginalPos;
        this.redGoal = redGoal;
        redWon = false;

        //Red Blue Initialization loop
        for (int i = 0; i < GameManager.EPISODE_LENGTH; i++) {
            //Red Blue State init
            blueStates[i] = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
            redStates[i] = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);

            //Red Blue Action and Log Prob init
            for (int j = 0; j < GameManager.TEAM_SIZE; j++) {
                blueActions[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
                blueLog_Probs[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
                redActions[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
                redLog_Probs[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
            }
        }
    }

    public void stepForward(NeuralNetwork[] blueAgents, NeuralNetwork[] blueCritics, NeuralNetwork[] redAgents, NeuralNetwork[] redCritics) {
        Rigidbody rb;
        MeshRenderer mesh;
        int stateIndex = 0;

        //Get Mask for time_step
        mask[time_step] = redWon || blueWon ? 0 : 1;

        //Get Rewards for State_time_step
        blueRewards[time_step] = blueReward();
        redRewards[time_step] = redReward();

        //Common State information for both blue and red
        NativeArray<double> curBlueState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        NativeArray<double> curRedState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            blueStates[time_step][stateIndex] = bluePlayers[i].transform.position.x;
            redStates[time_step][stateIndex] = bluePlayers[i].transform.position.x;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.x;
            curRedState[stateIndex] = bluePlayers[i].transform.position.x;
            stateIndex++;
            blueStates[time_step][stateIndex] = bluePlayers[i].transform.position.y;
            redStates[time_step][stateIndex] = bluePlayers[i].transform.position.y;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.y;
            curRedState[stateIndex] = bluePlayers[i].transform.position.y;
            stateIndex++;
            blueStates[time_step][stateIndex] = bluePlayers[i].transform.position.z;
            redStates[time_step][stateIndex] = bluePlayers[i].transform.position.z;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.z;
            curRedState[stateIndex] = bluePlayers[i].transform.position.z;
            stateIndex++;
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            blueStates[time_step][stateIndex] = rb.velocity.x;
            redStates[time_step][stateIndex] = rb.velocity.x;
            curBlueState[stateIndex] = rb.velocity.x;
            curRedState[stateIndex] = rb.velocity.x;
            stateIndex++;
            blueStates[time_step][stateIndex] = rb.velocity.y;
            redStates[time_step][stateIndex] = rb.velocity.y;
            curBlueState[stateIndex] = rb.velocity.y;
            curRedState[stateIndex] = rb.velocity.y;
            stateIndex++;
            blueStates[time_step][stateIndex] = rb.velocity.z;
            redStates[time_step][stateIndex] = rb.velocity.z;
            curBlueState[stateIndex] = rb.velocity.z;
            curRedState[stateIndex] = rb.velocity.z;
            stateIndex++;
            blueStates[time_step][stateIndex] = redPlayers[i].transform.position.x;
            redStates[time_step][stateIndex] = redPlayers[i].transform.position.x;
            curBlueState[stateIndex] = redPlayers[i].transform.position.x;
            curRedState[stateIndex] = redPlayers[i].transform.position.x;
            stateIndex++;
            blueStates[time_step][stateIndex] = redPlayers[i].transform.position.y;
            redStates[time_step][stateIndex] = redPlayers[i].transform.position.y;
            curBlueState[stateIndex] = redPlayers[i].transform.position.y;
            curRedState[stateIndex] = redPlayers[i].transform.position.y;
            stateIndex++;
            blueStates[time_step][stateIndex] = redPlayers[i].transform.position.z;
            redStates[time_step][stateIndex] = redPlayers[i].transform.position.z;
            curBlueState[stateIndex] = redPlayers[i].transform.position.z;
            curRedState[stateIndex] = redPlayers[i].transform.position.z;
            stateIndex++;
            rb = redPlayers[i].GetComponent<Rigidbody>();
            blueStates[time_step][stateIndex] = rb.velocity.x;
            redStates[time_step][stateIndex] = rb.velocity.x;
            curBlueState[stateIndex] = rb.velocity.x;
            curRedState[stateIndex] = rb.velocity.x;
            stateIndex++;
            blueStates[time_step][stateIndex] = rb.velocity.y;
            redStates[time_step][stateIndex] = rb.velocity.y;
            curBlueState[stateIndex] = rb.velocity.y;
            curRedState[stateIndex] = rb.velocity.y;
            stateIndex++;
            blueStates[time_step][stateIndex] = rb.velocity.z;
            redStates[time_step][stateIndex] = rb.velocity.z;
            curBlueState[stateIndex] = rb.velocity.z;
            curRedState[stateIndex] = rb.velocity.z;
            stateIndex++;
        }
        blueStates[time_step][stateIndex] = ball.transform.position.x;
        redStates[time_step][stateIndex] = ball.transform.position.x;
        curBlueState[stateIndex] = ball.transform.position.x;
        curRedState[stateIndex] = ball.transform.position.x;
        stateIndex++;
        blueStates[time_step][stateIndex] = ball.transform.position.y;
        redStates[time_step][stateIndex] = ball.transform.position.y;
        curBlueState[stateIndex] = ball.transform.position.y;
        curRedState[stateIndex] = ball.transform.position.y;
        stateIndex++;
        blueStates[time_step][stateIndex] = ball.transform.position.z;
        redStates[time_step][stateIndex] = ball.transform.position.z;
        curBlueState[stateIndex] = ball.transform.position.z;
        curRedState[stateIndex] = ball.transform.position.z;
        stateIndex++;
        rb = ball.GetComponent<Rigidbody>();
        blueStates[time_step][stateIndex] = rb.velocity.x;
        redStates[time_step][stateIndex] = rb.velocity.x;
        curBlueState[stateIndex] = rb.velocity.x;
        curRedState[stateIndex] = rb.velocity.x;
        stateIndex++;
        blueStates[time_step][stateIndex] = rb.velocity.y;
        redStates[time_step][stateIndex] = rb.velocity.y;
        curBlueState[stateIndex] = rb.velocity.y;
        curRedState[stateIndex] = rb.velocity.y;
        stateIndex++;
        blueStates[time_step][stateIndex] = rb.velocity.z;
        redStates[time_step][stateIndex] = rb.velocity.z;
        curBlueState[stateIndex] = rb.velocity.z;
        curRedState[stateIndex] = rb.velocity.z;
        stateIndex++;

        //Blue Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<MeshRenderer>();
        blueStates[time_step][stateIndex] = blueGoal.transform.position.x;
        curBlueState[stateIndex] = blueGoal.transform.position.x;
        stateIndex++;
        blueStates[time_step][stateIndex] = blueGoal.transform.position.y;
        curBlueState[stateIndex] = blueGoal.transform.position.y;
        stateIndex++;
        blueStates[time_step][stateIndex] = blueGoal.transform.position.z;
        curBlueState[stateIndex] = blueGoal.transform.position.z;
        stateIndex++;
        blueStates[time_step][stateIndex] = mesh.bounds.size.z;
        curBlueState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        blueStates[time_step][stateIndex] = mesh.bounds.size.y;
        curBlueState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        mesh = redGoal.GetComponent<MeshRenderer>();
        blueStates[time_step][stateIndex] = redGoal.transform.position.x;
        curBlueState[stateIndex] = redGoal.transform.position.x;
        stateIndex++;
        blueStates[time_step][stateIndex] = redGoal.transform.position.y;
        curBlueState[stateIndex] = redGoal.transform.position.y;
        stateIndex++;
        blueStates[time_step][stateIndex] = redGoal.transform.position.z;
        curBlueState[stateIndex] = redGoal.transform.position.z;
        stateIndex++;
        blueStates[time_step][stateIndex] = mesh.bounds.size.z;
        curBlueState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        blueStates[time_step][stateIndex] = mesh.bounds.size.y;
        curBlueState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Push back state index to fill last 10 spots again
        stateIndex -= 10;

        //Red Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        mesh = redGoal.GetComponent<MeshRenderer>();
        redStates[time_step][stateIndex] = redGoal.transform.position.x;
        curRedState[stateIndex] = redGoal.transform.position.x;
        stateIndex++;
        redStates[time_step][stateIndex] = redGoal.transform.position.y;
        curRedState[stateIndex] = redGoal.transform.position.y;
        stateIndex++;
        redStates[time_step][stateIndex] = redGoal.transform.position.z;
        curRedState[stateIndex] = redGoal.transform.position.z;
        stateIndex++;
        redStates[time_step][stateIndex] = mesh.bounds.size.z;
        curRedState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        redStates[time_step][stateIndex] = mesh.bounds.size.y;
        curRedState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<MeshRenderer>();
        redStates[time_step][stateIndex] = blueGoal.transform.position.x;
        curRedState[stateIndex] = blueGoal.transform.position.x;
        stateIndex++;
        redStates[time_step][stateIndex] = blueGoal.transform.position.y;
        curRedState[stateIndex] = blueGoal.transform.position.y;
        stateIndex++;
        redStates[time_step][stateIndex] = blueGoal.transform.position.z;
        curRedState[stateIndex] = blueGoal.transform.position.z;
        stateIndex++;
        redStates[time_step][stateIndex] = mesh.bounds.size.z;
        curRedState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        redStates[time_step][stateIndex] = mesh.bounds.size.y;
        curRedState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;
        NativeList<JobHandle> forwardJobHandles = new NativeList<JobHandle>(GameManager.TEAM_SIZE * 4, Allocator.Persistent);
        NativeArray<double>[,] actionDists = new NativeArray<double>[GameManager.TEAM_SIZE, 2];
        int id = 0;
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            actionDists[i, 0] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
            actionDists[i, 1] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);

            //Forward Step on Blue Agents + Critics
            forwardJobHandles.Add(blueAgents[i].Forward(curBlueState, ref actionDists[i, 0], id));
            blueValues[time_step, i] = new NativeArray<double>(1, Allocator.Persistent);
            forwardJobHandles.Add(blueCritics[i].Forward(curBlueState, ref blueValues[time_step, i], id));
            id++;

            //Forward Step on Red Agents + Critics
            forwardJobHandles.Add(redAgents[i].Forward(curRedState, ref actionDists[i, 1], id));
            redValues[time_step, i] = new NativeArray<double>(1, Allocator.Persistent);
            forwardJobHandles.Add(redCritics[i].Forward(curRedState, ref redValues[time_step, i], id));
            id++;
        }
        JobHandle.CompleteAll(forwardJobHandles);
        forwardJobHandles.Dispose();
        curBlueState.Dispose();
        curRedState.Dispose();

        //Apply Forces on Agents
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            //Blue Team Actions
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                blueActions[time_step, i][j] = GaussianDistribution.NextGaussian(actionDists[i, 0][j], blueAgents[i].log_std[j]);
                blueLog_Probs[time_step, i][j] = GaussianDistribution.log_prob(blueActions[time_step, i][j], actionDists[i, 0][j], blueAgents[i].log_std[j]);
            }
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            Vector3d force = Vector3d.Normalize(new Vector3d(blueActions[time_step, i][0], 0, blueActions[time_step, i][1]));
            force *= GameManager.MAX_SPEED*Sigmoid(blueActions[time_step, i][2]);
            //UnityEngine.Debug.Log(force);
            rb.velocity = new Vector3((float)force[0], (float)force[1], (float)force[2]);

            //Red Team Actions
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                redActions[time_step, i][j] = GaussianDistribution.NextGaussian(actionDists[i, 1][j], redAgents[i].log_std[j]);
                redLog_Probs[time_step, i][j] = GaussianDistribution.log_prob(redActions[time_step, i][j], actionDists[i, 1][j], redAgents[i].log_std[j]);
            }
            rb = redPlayers[i].GetComponent<Rigidbody>();
            force = Vector3d.Normalize(new Vector3d(redActions[time_step, i][0], 0, redActions[time_step, i][1]));
            force *= GameManager.MAX_SPEED*Sigmoid(redActions[time_step, i][2]);
            //UnityEngine.Debug.Log(force);
            rb.velocity = new Vector3((float)force[0], (float)force[1], (float)force[2]);
            actionDists[i, 0].Dispose();
            actionDists[i, 1].Dispose();
        }

        time_step++;
    }

    public void getNextValues(NeuralNetwork[] blueCritics, NeuralNetwork[] redCritics) {
        //Common State information for both blue and red
        NativeArray<double> curBlueState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        NativeArray<double> curRedState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        Rigidbody rb;
        MeshRenderer mesh;
        int stateIndex = 0;
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            curBlueState[stateIndex] = bluePlayers[i].transform.position.x;
            curRedState[stateIndex] = bluePlayers[i].transform.position.x;
            stateIndex++;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.y;
            curRedState[stateIndex] = bluePlayers[i].transform.position.y;
            stateIndex++;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.z;
            curRedState[stateIndex] = bluePlayers[i].transform.position.z;
            stateIndex++;
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            curBlueState[stateIndex] = rb.velocity.x;
            curRedState[stateIndex] = rb.velocity.x;
            stateIndex++;
            curBlueState[stateIndex] = rb.velocity.y;
            curRedState[stateIndex] = rb.velocity.y;
            stateIndex++;
            curBlueState[stateIndex] = rb.velocity.z;
            curRedState[stateIndex] = rb.velocity.z;
            stateIndex++;
            curBlueState[stateIndex] = redPlayers[i].transform.position.x;
            curRedState[stateIndex] = redPlayers[i].transform.position.x;
            stateIndex++;
            curBlueState[stateIndex] = redPlayers[i].transform.position.y;
            curRedState[stateIndex] = redPlayers[i].transform.position.y;
            stateIndex++;
            curBlueState[stateIndex] = redPlayers[i].transform.position.z;
            curRedState[stateIndex] = redPlayers[i].transform.position.z;
            stateIndex++;
            rb = redPlayers[i].GetComponent<Rigidbody>();
            curBlueState[stateIndex] = rb.velocity.x;
            curRedState[stateIndex] = rb.velocity.x;
            stateIndex++;
            curBlueState[stateIndex] = rb.velocity.y;
            curRedState[stateIndex] = rb.velocity.y;
            stateIndex++;
            curBlueState[stateIndex] = rb.velocity.z;
            curRedState[stateIndex] = rb.velocity.z;
            stateIndex++;
        }
        curBlueState[stateIndex] = ball.transform.position.x;
        curRedState[stateIndex] = ball.transform.position.x;
        stateIndex++;
        curBlueState[stateIndex] = ball.transform.position.y;
        curRedState[stateIndex] = ball.transform.position.y;
        stateIndex++;
        curBlueState[stateIndex] = ball.transform.position.z;
        curRedState[stateIndex] = ball.transform.position.z;
        stateIndex++;
        rb = ball.GetComponent<Rigidbody>();
        curBlueState[stateIndex] = rb.velocity.x;
        curRedState[stateIndex] = rb.velocity.x;
        stateIndex++;
        curBlueState[stateIndex] = rb.velocity.y;
        curRedState[stateIndex] = rb.velocity.y;
        stateIndex++;
        curBlueState[stateIndex] = rb.velocity.z;
        curRedState[stateIndex] = rb.velocity.z;
        stateIndex++;

        //Blue Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<MeshRenderer>();
        curBlueState[stateIndex] = blueGoal.transform.position.x;
        stateIndex++;
        curBlueState[stateIndex] = blueGoal.transform.position.y;
        stateIndex++;
        curBlueState[stateIndex] = blueGoal.transform.position.z;
        stateIndex++;
        curBlueState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        curBlueState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        mesh = redGoal.GetComponent<MeshRenderer>();
        curBlueState[stateIndex] = redGoal.transform.position.x;
        stateIndex++;
        curBlueState[stateIndex] = redGoal.transform.position.y;
        stateIndex++;
        curBlueState[stateIndex] = redGoal.transform.position.z;
        stateIndex++;
        curBlueState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        curBlueState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Push back state index to fill last 10 spots again
        stateIndex -= 10;

        //Red Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        mesh = redGoal.GetComponent<MeshRenderer>();
        curRedState[stateIndex] = redGoal.transform.position.x;
        stateIndex++;
        curRedState[stateIndex] = redGoal.transform.position.y;
        stateIndex++;
        curRedState[stateIndex] = redGoal.transform.position.z;
        stateIndex++;
        curRedState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        curRedState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<MeshRenderer>();
        curRedState[stateIndex] = blueGoal.transform.position.x;
        stateIndex++;
        curRedState[stateIndex] = blueGoal.transform.position.y;
        stateIndex++;
        curRedState[stateIndex] = blueGoal.transform.position.z;
        stateIndex++;
        curRedState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        curRedState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Get next values
        NativeList<JobHandle> forwardJobHandles = new NativeList<JobHandle>(GameManager.TEAM_SIZE * 2, Allocator.Persistent);
        int id = 0;
        for (int i = 0; i  < GameManager.TEAM_SIZE; i++) {
            blueNextVals[i] = new NativeArray<double>(1, Allocator.Persistent);
            redNextVals[i] = new NativeArray<double>(1, Allocator.Persistent);
            forwardJobHandles.Add(blueCritics[i].Forward(curBlueState, ref blueNextVals[i], id));
            id++;
            forwardJobHandles.Add(redCritics[i].Forward(curRedState, ref redNextVals[i], id));
            id++;
            //Stop agents due to end of episode
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            rb.velocity = new Vector3(0, 0, 0);
            rb = redPlayers[i].GetComponent<Rigidbody>();
            rb.velocity = new Vector3(0, 0, 0);
        }
        JobHandle.CompleteAll(forwardJobHandles);
        forwardJobHandles.Dispose();
        curBlueState.Dispose();
        curRedState.Dispose();
    }



    public void CalculateGAE() {
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            //Gather Blue and Red Agent i's rewards, values, next vals
            NativeArray<double> nativeBlueRewards = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            NativeArray<double> nativeBlueValues = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            double nextBlueVal = blueNextVals[i][0];
            NativeArray<double> nativeRedRewards = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            NativeArray<double> nativeRedValues = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            double nextRedVal = redNextVals[i][0];
            for (int j = 0; j < GameManager.EPISODE_LENGTH; j++) {
                nativeBlueRewards[j] = blueRewards[j, i];
                nativeBlueValues[j] = blueValues[j, i][0];
                nativeRedRewards[j] = redRewards[j, i];
                nativeRedValues[j] = redValues[j, i][0];
            }

            NativeArray<JobHandle> blueRedGAEJobs = new NativeArray<JobHandle>(2, Allocator.Persistent);

            CalculateGAEJob blueGAECalcJob = new CalculateGAEJob {
                rewards = nativeBlueRewards,
                values = nativeBlueValues,
                mask = mask,
                gamma = GameManager.GAMMA,
                lambda = GameManager.GAE_LAMBDA,
                next_value = nextBlueVal,
                numSteps = GameManager.EPISODE_LENGTH,
                agentInd = i,
                TEAM_SIZE = GameManager.TEAM_SIZE,
                returns = blueReturns,
                advantages = blueAdvantages,
            };

            CalculateGAEJob redGAECalcJob = new CalculateGAEJob {
                rewards = nativeRedRewards,
                values = nativeRedValues,
                mask = mask,
                gamma = GameManager.GAMMA,
                lambda = GameManager.GAE_LAMBDA,
                next_value = nextRedVal,
                numSteps = GameManager.EPISODE_LENGTH,
                agentInd = i,
                TEAM_SIZE = GameManager.TEAM_SIZE,
                returns = redReturns,
                advantages = redAdvantages,
            };
            blueRedGAEJobs[0] = blueGAECalcJob.Schedule();
            blueRedGAEJobs[1] = redGAECalcJob.Schedule();
            JobHandle.CompleteAll(blueRedGAEJobs);

            //Dispose job specific Native Arrays
            blueRedGAEJobs.Dispose();
            nativeBlueRewards.Dispose();
            nativeBlueValues.Dispose();
            nativeRedRewards.Dispose();
            nativeRedValues.Dispose();

        }
    }
    
    [BurstCompile]
    public double blueReward() {
        if (!redWon && !blueWon) {
            return math.abs(ball.transform.position.x-redGoal.transform.position.x)/GameManager.FIELD_LENGTH;
        } else if (redWon) {
            return -1000;
        } else {
            return 1000;
        }
    }

    [BurstCompile]
    public double redReward() {
        if (!redWon && !blueWon) {
            return math.abs(ball.transform.position.x-blueGoal.transform.position.x)/GameManager.FIELD_LENGTH;
        } else if (blueWon) {
            return -1000;
        } else {
            return 1000;
        }
    }

    [BurstCompile]
    public void Dispose() {
        mask.Dispose();
        blueReturns.Dispose();
        blueAdvantages.Dispose();
        redReturns.Dispose();
        redAdvantages.Dispose();
    }

    [BurstCompile]
    private double Sigmoid(double x) {
        return 1 / (math.exp(-x) + 1);
    }

    [BurstCompile]
    public void resetEnv() {
        blueWon = false;
        redWon = false;
        ball.transform.position = ballOriginalPos;
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            bluePlayers[i].transform.position = blueOriginalPos[i];
            redPlayers[i].transform.position = redOriginalPos[i];
        }
        for (int i = 0; i < GameManager.EPISODE_LENGTH; i++) {
            blueStates[i].Dispose();
            redStates[i].Dispose();
            for (int j = 0; j < GameManager.TEAM_SIZE; j++) {
                blueActions[i, j].Dispose();
                blueLog_Probs[i, j].Dispose();
                redActions[i, j].Dispose();
                redLog_Probs[i, j].Dispose();
            }
        }
    }
}
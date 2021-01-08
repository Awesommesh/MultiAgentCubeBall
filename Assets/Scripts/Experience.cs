using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using UnityEngine;
using Unity.Burst;
using System;
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
    bool blueGotGoalReward;
    public double avgBlueReward;

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
    bool redGotGoalReward;
    public double avgRedReward;

    //General
    GameObject ball;
    Vector3 ballOriginalPos;
    public NativeArray<int> mask;
    int time_step = 0;
    GoalDetector ballGoal; 
    double FIELD_LENGTH;
    const double FIELD_WIDTH = 30;
    double goalWidth = 18;
    double goalHeight = 8;
    Unity.Mathematics.Random customSampler;
    public Experience (GameObject ball, GameObject blueGoal, GameObject redGoal, GameObject[] blueTeam, GameObject[] redTeam, 
        Vector3 ballOriginalPos, Vector3[] blueOriginalPos, Vector3[] redOriginalPos, uint samplerSeed) {
        //General Initialization
        time_step = 0;
        this.ball = ball;
        ballGoal = ball.GetComponent<GoalDetector>();
        ballGoal.init();
        this.ballOriginalPos = ballOriginalPos;
        mask = new NativeArray<int>(GameManager.EPISODE_LENGTH, Allocator.Persistent);
        FIELD_LENGTH = math.abs(blueGoal.transform.localPosition.x - redGoal.transform.localPosition.x);
        customSampler = new Unity.Mathematics.Random(samplerSeed);

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
        blueGotGoalReward = false;
        avgBlueReward = 0;

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
        redGotGoalReward = false;
        avgRedReward = 0;

        //Red Blue Initialization loops
        for (int i = 0; i < GameManager.EPISODE_LENGTH; i++) {
            //Red Blue State init
            blueStates[i] = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
            redStates[i] = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);

            //Red Blue Action and Log Prob init
            for (int j = 0; j < GameManager.TEAM_SIZE; j++) {
                blueActions[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
                blueLog_Probs[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
                blueValues[i, j] = new NativeArray<double>(1, Allocator.Persistent);
                redActions[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
                redLog_Probs[i, j] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
                redValues[i, j] = new NativeArray<double>(1, Allocator.Persistent);
            }
        }
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            blueNextVals[i] = new NativeArray<double>(1, Allocator.Persistent);
            redNextVals[i] = new NativeArray<double>(1, Allocator.Persistent);
        }
    }

    public void stepForward(NeuralNetwork[] blueAgents, NeuralNetwork[] blueCritics, NeuralNetwork[] redAgents, NeuralNetwork[] redCritics) {
        Rigidbody rb;
        int stateIndex = 0;
        ballGoal.checkGoalScored();
        if (ballGoal.blueWon || ballGoal.redWon) {
            if (!redGotGoalReward && !blueGotGoalReward) {
                blueWon = ballGoal.blueWon;
                redWon = ballGoal.redWon;
            }
            ballGoal.blueWon = false;
            ballGoal.redWon = false;
            resetlocalPositions();
        }

        //Get Rewards for State_time_step
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            blueRewards[time_step, i] = blueReward(i);
            redRewards[time_step, i] = redReward(i);
            avgBlueReward += blueRewards[time_step, i];
            avgRedReward += redRewards[time_step, i];
        }

        //Get Mask for time_step
        mask[time_step] = blueGotGoalReward && redGotGoalReward ? 0 : 1;

        //Common State information for both blue and red
        NativeArray<double> curBlueState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        NativeArray<double> curRedState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            blueStates[time_step][stateIndex] = bluePlayers[i].transform.localPosition.x;
            redStates[time_step][stateIndex] = bluePlayers[i].transform.localPosition.x;
            curBlueState[stateIndex] = bluePlayers[i].transform.localPosition.x;
            curRedState[stateIndex] = bluePlayers[i].transform.localPosition.x;
            stateIndex++;
            blueStates[time_step][stateIndex] = bluePlayers[i].transform.localPosition.y;
            redStates[time_step][stateIndex] = bluePlayers[i].transform.localPosition.y;
            curBlueState[stateIndex] = bluePlayers[i].transform.localPosition.y;
            curRedState[stateIndex] = bluePlayers[i].transform.localPosition.y;
            stateIndex++;
            blueStates[time_step][stateIndex] = bluePlayers[i].transform.localPosition.z;
            redStates[time_step][stateIndex] = bluePlayers[i].transform.localPosition.z;
            curBlueState[stateIndex] = bluePlayers[i].transform.localPosition.z;
            curRedState[stateIndex] = bluePlayers[i].transform.localPosition.z;
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
            blueStates[time_step][stateIndex] = redPlayers[i].transform.localPosition.x;
            redStates[time_step][stateIndex] = redPlayers[i].transform.localPosition.x;
            curBlueState[stateIndex] = redPlayers[i].transform.localPosition.x;
            curRedState[stateIndex] = redPlayers[i].transform.localPosition.x;
            stateIndex++;
            blueStates[time_step][stateIndex] = redPlayers[i].transform.localPosition.y;
            redStates[time_step][stateIndex] = redPlayers[i].transform.localPosition.y;
            curBlueState[stateIndex] = redPlayers[i].transform.localPosition.y;
            curRedState[stateIndex] = redPlayers[i].transform.localPosition.y;
            stateIndex++;
            blueStates[time_step][stateIndex] = redPlayers[i].transform.localPosition.z;
            redStates[time_step][stateIndex] = redPlayers[i].transform.localPosition.z;
            curBlueState[stateIndex] = redPlayers[i].transform.localPosition.z;
            curRedState[stateIndex] = redPlayers[i].transform.localPosition.z;
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
        blueStates[time_step][stateIndex] = ball.transform.localPosition.x;
        redStates[time_step][stateIndex] = ball.transform.localPosition.x;
        curBlueState[stateIndex] = ball.transform.localPosition.x;
        curRedState[stateIndex] = ball.transform.localPosition.x;
        stateIndex++;
        blueStates[time_step][stateIndex] = ball.transform.localPosition.y;
        redStates[time_step][stateIndex] = ball.transform.localPosition.y;
        curBlueState[stateIndex] = ball.transform.localPosition.y;
        curRedState[stateIndex] = ball.transform.localPosition.y;
        stateIndex++;
        blueStates[time_step][stateIndex] = ball.transform.localPosition.z;
        redStates[time_step][stateIndex] = ball.transform.localPosition.z;
        curBlueState[stateIndex] = ball.transform.localPosition.z;
        curRedState[stateIndex] = ball.transform.localPosition.z;
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
        blueStates[time_step][stateIndex] = blueGoal.transform.localPosition.x;
        curBlueState[stateIndex] = blueGoal.transform.localPosition.x;
        stateIndex++;
        blueStates[time_step][stateIndex] = blueGoal.transform.localPosition.y;
        curBlueState[stateIndex] = blueGoal.transform.localPosition.y;
        stateIndex++;
        blueStates[time_step][stateIndex] = blueGoal.transform.localPosition.z;
        curBlueState[stateIndex] = blueGoal.transform.localPosition.z;
        stateIndex++;
        blueStates[time_step][stateIndex] = goalWidth;
        curBlueState[stateIndex] = goalWidth;
        stateIndex++;
        blueStates[time_step][stateIndex] = goalHeight;
        curBlueState[stateIndex] = goalHeight;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        blueStates[time_step][stateIndex] = redGoal.transform.localPosition.x;
        curBlueState[stateIndex] = redGoal.transform.localPosition.x;
        stateIndex++;
        blueStates[time_step][stateIndex] = redGoal.transform.localPosition.y;
        curBlueState[stateIndex] = redGoal.transform.localPosition.y;
        stateIndex++;
        blueStates[time_step][stateIndex] = redGoal.transform.localPosition.z;
        curBlueState[stateIndex] = redGoal.transform.localPosition.z;
        stateIndex++;
        blueStates[time_step][stateIndex] = goalWidth;
        curBlueState[stateIndex] = goalWidth;
        stateIndex++;
        blueStates[time_step][stateIndex] = goalHeight;
        curBlueState[stateIndex] = goalHeight;
        stateIndex++;

        //Push back state index to fill last 10 spots again
        stateIndex -= 10;

        //Red Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        redStates[time_step][stateIndex] = redGoal.transform.localPosition.x;
        curRedState[stateIndex] = redGoal.transform.localPosition.x;
        stateIndex++;
        redStates[time_step][stateIndex] = redGoal.transform.localPosition.y;
        curRedState[stateIndex] = redGoal.transform.localPosition.y;
        stateIndex++;
        redStates[time_step][stateIndex] = redGoal.transform.localPosition.z;
        curRedState[stateIndex] = redGoal.transform.localPosition.z;
        stateIndex++;
        redStates[time_step][stateIndex] = goalWidth;
        curRedState[stateIndex] = goalWidth;
        stateIndex++;
        redStates[time_step][stateIndex] = goalHeight;
        curRedState[stateIndex] = goalHeight;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        redStates[time_step][stateIndex] = blueGoal.transform.localPosition.x;
        curRedState[stateIndex] = blueGoal.transform.localPosition.x;
        stateIndex++;
        redStates[time_step][stateIndex] = blueGoal.transform.localPosition.y;
        curRedState[stateIndex] = blueGoal.transform.localPosition.y;
        stateIndex++;
        redStates[time_step][stateIndex] = blueGoal.transform.localPosition.z;
        curRedState[stateIndex] = blueGoal.transform.localPosition.z;
        stateIndex++;
        redStates[time_step][stateIndex] = goalWidth;
        curRedState[stateIndex] = goalWidth;
        stateIndex++;
        redStates[time_step][stateIndex] = goalHeight;
        curRedState[stateIndex] = goalHeight;
        stateIndex++;
        NativeList<JobHandle> forwardJobHandles = new NativeList<JobHandle>(GameManager.TEAM_SIZE * 4, Allocator.Persistent);
        NativeArray<double>[,] actionDists = new NativeArray<double>[GameManager.TEAM_SIZE, 2];
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            actionDists[i, 0] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);
            actionDists[i, 1] = new NativeArray<double>(GameManager.NUM_ACTIONS, Allocator.Persistent);

            //Forward Step on Blue Agents + Critics
            forwardJobHandles.Add(blueAgents[i].Forward(curBlueState, 1, actionDists[i, 0], 0));
            forwardJobHandles.Add(blueCritics[i].Forward(curBlueState, 1, blueValues[time_step, i], 0));

            //Forward Step on Red Agents + Critics
            forwardJobHandles.Add(redAgents[i].Forward(curRedState, 1, actionDists[i, 1], 1));
            forwardJobHandles.Add(redCritics[i].Forward(curRedState, 1, redValues[time_step, i], 1));
        }
        JobHandle.CompleteAll(forwardJobHandles);
        forwardJobHandles.Dispose();
        curBlueState.Dispose();
        curRedState.Dispose();

        //Apply Forces on Agents
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            //Blue Team Actions
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                blueActions[time_step, i][j] = GaussianDistribution.NextGaussian(actionDists[i, 0][j], blueAgents[i].std[j], customSampler);
                blueLog_Probs[time_step, i][j] = GaussianDistribution.log_prob(blueActions[time_step, i][j], actionDists[i, 0][j], blueAgents[i].std[j]);
            }
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                if (Math.Abs(actionDists[i, 0][j]) > 100) {
                    Debug.Log("wtf:" + actionDists[i, 0][j]);
                }
            }
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            Vector3d force = Vector3d.Normalize(new Vector3d(blueActions[time_step, i][0], 0, blueActions[time_step, i][1]));
            force *= GameManager.MAX_SPEED*Sigmoid(blueActions[time_step, i][2]);
            rb.velocity = new Vector3((float)force[0], (float)force[1], (float)force[2]);

            //Red Team Actions
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                redActions[time_step, i][j] = GaussianDistribution.NextGaussian(actionDists[i, 1][j], redAgents[i].std[j], customSampler);
                redLog_Probs[time_step, i][j] = GaussianDistribution.log_prob(redActions[time_step, i][j], actionDists[i, 1][j], redAgents[i].std[j]);
            }
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                if (Math.Abs(actionDists[i, 0][j]) > 100) {
                    Debug.Log("wtf1:" + actionDists[i, 1][j]);
                }
            }
            rb = redPlayers[i].GetComponent<Rigidbody>();
            force = Vector3d.Normalize(new Vector3d(redActions[time_step, i][0], 0, redActions[time_step, i][1]));
            force *= GameManager.MAX_SPEED*Sigmoid(redActions[time_step, i][2]);
            rb.velocity = new Vector3((float)force[0], (float)force[1], (float)force[2]);
            actionDists[i, 0].Dispose();
            actionDists[i, 1].Dispose();
        }

        time_step++;
    }

    public void getNextValues(NeuralNetwork[] blueCritics, NeuralNetwork[] redCritics) {
        //Average total blue and red reward for evaluation
        avgBlueReward /= (GameManager.EPISODE_LENGTH*GameManager.TEAM_SIZE);
        avgRedReward /= (GameManager.EPISODE_LENGTH*GameManager.TEAM_SIZE);
        //Common State information for both blue and red
        NativeArray<double> curBlueState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        NativeArray<double> curRedState = new NativeArray<double>(GameManager.STATE_SIZE, Allocator.Persistent);
        Rigidbody rb;
        int stateIndex = 0;
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            curBlueState[stateIndex] = bluePlayers[i].transform.localPosition.x;
            curRedState[stateIndex] = bluePlayers[i].transform.localPosition.x;
            stateIndex++;
            curBlueState[stateIndex] = bluePlayers[i].transform.localPosition.y;
            curRedState[stateIndex] = bluePlayers[i].transform.localPosition.y;
            stateIndex++;
            curBlueState[stateIndex] = bluePlayers[i].transform.localPosition.z;
            curRedState[stateIndex] = bluePlayers[i].transform.localPosition.z;
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
            curBlueState[stateIndex] = redPlayers[i].transform.localPosition.x;
            curRedState[stateIndex] = redPlayers[i].transform.localPosition.x;
            stateIndex++;
            curBlueState[stateIndex] = redPlayers[i].transform.localPosition.y;
            curRedState[stateIndex] = redPlayers[i].transform.localPosition.y;
            stateIndex++;
            curBlueState[stateIndex] = redPlayers[i].transform.localPosition.z;
            curRedState[stateIndex] = redPlayers[i].transform.localPosition.z;
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
        curBlueState[stateIndex] = ball.transform.localPosition.x;
        curRedState[stateIndex] = ball.transform.localPosition.x;
        stateIndex++;
        curBlueState[stateIndex] = ball.transform.localPosition.y;
        curRedState[stateIndex] = ball.transform.localPosition.y;
        stateIndex++;
        curBlueState[stateIndex] = ball.transform.localPosition.z;
        curRedState[stateIndex] = ball.transform.localPosition.z;
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
        curBlueState[stateIndex] = blueGoal.transform.localPosition.x;
        stateIndex++;
        curBlueState[stateIndex] = blueGoal.transform.localPosition.y;
        stateIndex++;
        curBlueState[stateIndex] = blueGoal.transform.localPosition.z;
        stateIndex++;
        curBlueState[stateIndex] = goalWidth;
        stateIndex++;
        curBlueState[stateIndex] = goalHeight;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        curBlueState[stateIndex] = redGoal.transform.localPosition.x;
        stateIndex++;
        curBlueState[stateIndex] = redGoal.transform.localPosition.y;
        stateIndex++;
        curBlueState[stateIndex] = redGoal.transform.localPosition.z;
        stateIndex++;
        curBlueState[stateIndex] = goalWidth;
        stateIndex++;
        curBlueState[stateIndex] = goalHeight;
        stateIndex++;

        //Push back state index to fill last 10 spots again
        stateIndex -= 10;

        //Red Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        curRedState[stateIndex] = redGoal.transform.localPosition.x;
        stateIndex++;
        curRedState[stateIndex] = redGoal.transform.localPosition.y;
        stateIndex++;
        curRedState[stateIndex] = redGoal.transform.localPosition.z;
        stateIndex++;
        curRedState[stateIndex] = goalWidth;
        stateIndex++;
        curRedState[stateIndex] = goalHeight;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        curRedState[stateIndex] = blueGoal.transform.localPosition.x;
        stateIndex++;
        curRedState[stateIndex] = blueGoal.transform.localPosition.y;
        stateIndex++;
        curRedState[stateIndex] = blueGoal.transform.localPosition.z;
        stateIndex++;
        curRedState[stateIndex] = goalWidth;
        stateIndex++;
        curRedState[stateIndex] = goalHeight;
        stateIndex++;

        //Get next values
        NativeList<JobHandle> forwardJobHandles = new NativeList<JobHandle>(GameManager.TEAM_SIZE * 2, Allocator.Persistent);
        for (int i = 0; i  < GameManager.TEAM_SIZE; i++) {
            forwardJobHandles.Add(blueCritics[i].Forward(curBlueState, 1, blueNextVals[i], 0));
            forwardJobHandles.Add(redCritics[i].Forward(curRedState, 1, redNextVals[i], 1));
            //Stop agents due to end of episode
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            rb.velocity = new Vector3(0, 0, 0);
            rb = redPlayers[i].GetComponent<Rigidbody>();
            rb.velocity = new Vector3(0, 0, 0);
        }
        //Stop ball due to end of episode
        rb = ball.GetComponent<Rigidbody>();
        rb.velocity = new Vector3(0, 0, 0);

        JobHandle.CompleteAll(forwardJobHandles);
        forwardJobHandles.Dispose();
        curBlueState.Dispose();
        curRedState.Dispose();
    }

    [BurstCompile]
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
                EPSILON = GameManager.EPSILON
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
                EPSILON = GameManager.EPSILON
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
    public double blueReward(int playerInd) {
        if (blueGotGoalReward) {
            //Avg Distance of players to ball
            double totDist = 0;
            double xDiff = math.abs(ball.transform.localPosition.x-bluePlayers[playerInd].transform.localPosition.x);
            double zDiff = math.abs(ball.transform.localPosition.z-bluePlayers[playerInd].transform.localPosition.z);
            totDist += 1 - ((2*xDiff)/FIELD_LENGTH);
            totDist += 1 - ((2*zDiff)/FIELD_WIDTH);
            totDist /= 2;
            if (totDist < GameManager.EPSILON) {
                totDist = 0;
            }
            return totDist;
            //return 0.5-(math.abs(ball.transform.localPosition.x-redGoal.transform.localPosition.x)/FIELD_LENGTH);
        }
        if (!redWon && !blueWon) {
            //Avg Distance of players to ball
            double totDist = 0;
            double xDiff = math.abs(ball.transform.localPosition.x-bluePlayers[playerInd].transform.localPosition.x);
            double zDiff = math.abs(ball.transform.localPosition.z-bluePlayers[playerInd].transform.localPosition.z);
            totDist += 1 - ((2*xDiff)/FIELD_LENGTH);
            totDist += 1 - ((2*zDiff)/FIELD_WIDTH);
            totDist /= 2;
            if (totDist < GameManager.EPSILON) {
                totDist = 0;
            }
            return totDist;
            //Distance of ball to goal
            //return 0.5-(math.abs(ball.transform.localPosition.x-redGoal.transform.localPosition.x)/FIELD_LENGTH);
        } else if (redWon) {
            blueGotGoalReward = true;
            return -1000;
        } else {
            blueGotGoalReward = true;
            return 1000;
        }
    }

    [BurstCompile]
    public double redReward(int playerInd) {
        if (redGotGoalReward) {
            //Avg Distance of players to ball
            double totDist = 0;
            double xDiff = math.abs(ball.transform.localPosition.x-redPlayers[playerInd].transform.localPosition.x);
            double zDiff = math.abs(ball.transform.localPosition.z-redPlayers[playerInd].transform.localPosition.z);
            totDist += 1 - ((2*xDiff)/FIELD_LENGTH);
            totDist += 1 - ((2*zDiff)/FIELD_WIDTH);
            totDist /= 2;
            if (totDist < GameManager.EPSILON) {
                totDist = 0;
            }
            return totDist;
            //return 0.5-(math.abs(ball.transform.localPosition.x-blueGoal.transform.localPosition.x)/FIELD_LENGTH);
        }
        if (!redWon && !blueWon) {
            //Avg Distance of players to ball
            double totDist = 0;
            double xDiff = math.abs(ball.transform.localPosition.x-redPlayers[playerInd].transform.localPosition.x);
            double zDiff = math.abs(ball.transform.localPosition.z-redPlayers[playerInd].transform.localPosition.z);
            totDist += 1 - ((2*xDiff)/FIELD_LENGTH);
            totDist += 1 - ((2*zDiff)/FIELD_WIDTH);
            totDist /= 2;
            if (totDist < GameManager.EPSILON) {
                totDist = 0;
            }
            return totDist;
            //Distance of ball to goal
            //return 0.5-(math.abs(ball.transform.localPosition.x-blueGoal.transform.localPosition.x)/FIELD_LENGTH);
        } else if (blueWon) {
            redGotGoalReward = true;
            return -1000;
        } else {
            redGotGoalReward = true;
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
        for (int i = 0; i < GameManager.EPISODE_LENGTH; i++) {
            blueStates[i].Dispose();
            redStates[i].Dispose();
            for (int j = 0; j < GameManager.TEAM_SIZE; j++) {
                blueActions[i, j].Dispose();
                blueLog_Probs[i, j].Dispose();
                blueValues[i, j].Dispose();
                redActions[i, j].Dispose();
                redLog_Probs[i, j].Dispose();
                redValues[i, j].Dispose();
            }
        }
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            blueNextVals[i].Dispose();
            redNextVals[i].Dispose();
        }
    }

    [BurstCompile]
    private double Sigmoid(double x) {
        return 1 / (math.exp(-x) + 1);
    }

    [BurstCompile]
    public void resetEnv() {
        blueWon = false;
        redWon = false;
        resetlocalPositions();
        time_step = 0;
    }

    [BurstCompile]
    void resetlocalPositions() {
        ball.transform.position = ballOriginalPos;
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            bluePlayers[i].transform.position = blueOriginalPos[i];
            redPlayers[i].transform.position = redOriginalPos[i];
        }
    }
}
using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using UnityEngine;
using Unity.Burst;
public class Experience {
    //Blue Team
    public NDArray blueRewards;
    public NDArray blueValues;
    public NDArray blueStates;
    public NDArray blueActions;
    public NDArray blueLog_Probs;
    public NativeArray<double> blueNextVals;
    public NativeArray<double> blueReturns;
    public NativeArray<double> blueAdvantages;
    GameObject[] bluePlayers;
    public GameObject blueGoal;
    public bool blueWon;

    //Red Team   
    public NDArray redRewards;
    public NDArray redValues;
    public NDArray redStates;
    public NDArray redActions;
    public NDArray redLog_Probs;
    public NativeArray<double> redNextVals;
    public NativeArray<double> redReturns;
    public NativeArray<double> redAdvantages;
    GameObject[] redPlayers;
    public GameObject redGoal;
    public bool redWon;

    //General
    GameObject ball;
    public NativeArray<int> mask;
    int time_step = 0;
    
    public Experience (GameObject ball, GameObject blueGoal, GameObject redGoal, GameObject[] blueTeam, GameObject[] redTeam) {
        //General Initialization
        time_step = 0;
        this.ball = ball;
        mask = new NativeArray<int>(GameManager.EPISODE_LENGTH, Allocator.Persistent);

        //Blue initialization
        blueRewards = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE);
        blueValues = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE);
        blueStates = new NDArray(GameManager.EPISODE_LENGTH, GameManager.STATE_SIZE);
        blueActions = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE, GameManager.NUM_ACTIONS);
        blueLog_Probs = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE, GameManager.NUM_ACTIONS);
        blueNextVals = new NativeArray<double>(GameManager.TEAM_SIZE, Allocator.Persistent);
        blueReturns = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        blueAdvantages = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        bluePlayers = blueTeam;
        this.blueGoal = blueGoal;
        blueWon = false;

        //Red Initialization
        redRewards = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE);
        redValues = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE);
        redStates = new NDArray(GameManager.EPISODE_LENGTH, GameManager.STATE_SIZE);
        redActions = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE, GameManager.NUM_ACTIONS);
        redLog_Probs = new NDArray(GameManager.EPISODE_LENGTH, GameManager.TEAM_SIZE, GameManager.NUM_ACTIONS);
        redNextVals = new NativeArray<double>(GameManager.TEAM_SIZE, Allocator.Persistent);
        redReturns = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        redAdvantages = new NativeArray<double>(GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE, Allocator.Persistent);
        redPlayers = redTeam;
        this.redGoal = redGoal;
        redWon = false;
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
        NDArray curBlueState = new NDArray(GameManager.STATE_SIZE);
        NDArray curRedState = new NDArray(GameManager.STATE_SIZE);;
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            blueStates[time_step, stateIndex] = bluePlayers[i].transform.position.x;
            redStates[time_step, stateIndex] = bluePlayers[i].transform.position.x;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.x;
            curRedState[stateIndex] = bluePlayers[i].transform.position.x;
            stateIndex++;
            blueStates[time_step, stateIndex] = bluePlayers[i].transform.position.y;
            redStates[time_step, stateIndex] = bluePlayers[i].transform.position.y;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.y;
            curRedState[stateIndex] = bluePlayers[i].transform.position.y;
            stateIndex++;
            blueStates[time_step, stateIndex] = bluePlayers[i].transform.position.z;
            redStates[time_step, stateIndex] = bluePlayers[i].transform.position.z;
            curBlueState[stateIndex] = bluePlayers[i].transform.position.z;
            curRedState[stateIndex] = bluePlayers[i].transform.position.z;
            stateIndex++;
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            blueStates[time_step, stateIndex] = rb.velocity.x;
            redStates[time_step, stateIndex] = rb.velocity.x;
            curBlueState[stateIndex] = rb.velocity.x;
            curRedState[stateIndex] = rb.velocity.x;
            stateIndex++;
            blueStates[time_step, stateIndex] = rb.velocity.y;
            redStates[time_step, stateIndex] = rb.velocity.y;
            curBlueState[stateIndex] = rb.velocity.y;
            curRedState[stateIndex] = rb.velocity.y;
            stateIndex++;
            blueStates[time_step, stateIndex] = rb.velocity.z;
            redStates[time_step, stateIndex] = rb.velocity.z;
            curBlueState[stateIndex] = rb.velocity.z;
            curRedState[stateIndex] = rb.velocity.z;
            stateIndex++;
            blueStates[time_step, stateIndex] = redPlayers[i].transform.position.x;
            redStates[time_step, stateIndex] = redPlayers[i].transform.position.x;
            curBlueState[stateIndex] = redPlayers[i].transform.position.x;
            curRedState[stateIndex] = redPlayers[i].transform.position.x;
            stateIndex++;
            blueStates[time_step, stateIndex] = redPlayers[i].transform.position.y;
            redStates[time_step, stateIndex] = redPlayers[i].transform.position.y;
            curBlueState[stateIndex] = redPlayers[i].transform.position.y;
            curRedState[stateIndex] = redPlayers[i].transform.position.y;
            stateIndex++;
            blueStates[time_step, stateIndex] = redPlayers[i].transform.position.z;
            redStates[time_step, stateIndex] = redPlayers[i].transform.position.z;
            curBlueState[stateIndex] = redPlayers[i].transform.position.z;
            curRedState[stateIndex] = redPlayers[i].transform.position.z;
            stateIndex++;
            rb = redPlayers[i].GetComponent<Rigidbody>();
            blueStates[time_step, stateIndex] = rb.velocity.x;
            redStates[time_step, stateIndex] = rb.velocity.x;
            curBlueState[stateIndex] = rb.velocity.x;
            curRedState[stateIndex] = rb.velocity.x;
            stateIndex++;
            blueStates[time_step, stateIndex] = rb.velocity.y;
            redStates[time_step, stateIndex] = rb.velocity.y;
            curBlueState[stateIndex] = rb.velocity.y;
            curRedState[stateIndex] = rb.velocity.y;
            stateIndex++;
            blueStates[time_step, stateIndex] = rb.velocity.z;
            redStates[time_step, stateIndex] = rb.velocity.z;
            curBlueState[stateIndex] = rb.velocity.z;
            curRedState[stateIndex] = rb.velocity.z;
            stateIndex++;
        }
        blueStates[time_step, stateIndex] = ball.transform.position.x;
        redStates[time_step, stateIndex] = ball.transform.position.x;
        curBlueState[stateIndex] = ball.transform.position.x;
        curRedState[stateIndex] = ball.transform.position.x;
        stateIndex++;
        blueStates[time_step, stateIndex] = ball.transform.position.y;
        redStates[time_step, stateIndex] = ball.transform.position.y;
        curBlueState[stateIndex] = ball.transform.position.y;
        curRedState[stateIndex] = ball.transform.position.y;
        stateIndex++;
        blueStates[time_step, stateIndex] = ball.transform.position.z;
        redStates[time_step, stateIndex] = ball.transform.position.z;
        curBlueState[stateIndex] = ball.transform.position.z;
        curRedState[stateIndex] = ball.transform.position.z;
        stateIndex++;
        rb = ball.GetComponent<Rigidbody>();
        blueStates[time_step, stateIndex] = rb.velocity.x;
        redStates[time_step, stateIndex] = rb.velocity.x;
        curBlueState[stateIndex] = rb.velocity.x;
        curRedState[stateIndex] = rb.velocity.x;
        stateIndex++;
        blueStates[time_step, stateIndex] = rb.velocity.y;
        redStates[time_step, stateIndex] = rb.velocity.y;
        curBlueState[stateIndex] = rb.velocity.y;
        curRedState[stateIndex] = rb.velocity.y;
        stateIndex++;
        blueStates[time_step, stateIndex] = rb.velocity.z;
        redStates[time_step, stateIndex] = rb.velocity.z;
        curBlueState[stateIndex] = rb.velocity.z;
        curRedState[stateIndex] = rb.velocity.z;
        stateIndex++;

        //Blue Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<MeshRenderer>();
        blueStates[time_step, stateIndex] = blueGoal.transform.position.x;
        curBlueState[stateIndex] = blueGoal.transform.position.x;
        stateIndex++;
        blueStates[time_step, stateIndex] = blueGoal.transform.position.y;
        curBlueState[stateIndex] = blueGoal.transform.position.y;
        stateIndex++;
        blueStates[time_step, stateIndex] = blueGoal.transform.position.z;
        curBlueState[stateIndex] = blueGoal.transform.position.z;
        stateIndex++;
        blueStates[time_step, stateIndex] = mesh.bounds.size.z;
        curBlueState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        blueStates[time_step, stateIndex] = mesh.bounds.size.y;
        curBlueState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        mesh = redGoal.GetComponent<MeshRenderer>();
        blueStates[time_step, stateIndex] = redGoal.transform.position.x;
        curBlueState[stateIndex] = redGoal.transform.position.x;
        stateIndex++;
        blueStates[time_step, stateIndex] = redGoal.transform.position.y;
        curBlueState[stateIndex] = redGoal.transform.position.y;
        stateIndex++;
        blueStates[time_step, stateIndex] = redGoal.transform.position.z;
        curBlueState[stateIndex] = redGoal.transform.position.z;
        stateIndex++;
        blueStates[time_step, stateIndex] = mesh.bounds.size.z;
        curBlueState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        blueStates[time_step, stateIndex] = mesh.bounds.size.y;
        curBlueState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Push back state index to fill last 10 spots again
        stateIndex -= 10;

        //Red Team Specific State Setup
        //Friendly goal (!!! z is width and y is height !!!)
        mesh = redGoal.GetComponent<MeshRenderer>();
        redStates[time_step, stateIndex] = redGoal.transform.position.x;
        curRedState[stateIndex] = redGoal.transform.position.x;
        stateIndex++;
        redStates[time_step, stateIndex] = redGoal.transform.position.y;
        curRedState[stateIndex] = redGoal.transform.position.y;
        stateIndex++;
        redStates[time_step, stateIndex] = redGoal.transform.position.z;
        curRedState[stateIndex] = redGoal.transform.position.z;
        stateIndex++;
        redStates[time_step, stateIndex] = mesh.bounds.size.z;
        curRedState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        redStates[time_step, stateIndex] = mesh.bounds.size.y;
        curRedState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<MeshRenderer>();
        redStates[time_step, stateIndex] = blueGoal.transform.position.x;
        curRedState[stateIndex] = blueGoal.transform.position.x;
        stateIndex++;
        redStates[time_step, stateIndex] = blueGoal.transform.position.y;
        curRedState[stateIndex] = blueGoal.transform.position.y;
        stateIndex++;
        redStates[time_step, stateIndex] = blueGoal.transform.position.z;
        curRedState[stateIndex] = blueGoal.transform.position.z;
        stateIndex++;
        redStates[time_step, stateIndex] = mesh.bounds.size.z;
        curRedState[stateIndex] = mesh.bounds.size.z;
        stateIndex++;
        redStates[time_step, stateIndex] = mesh.bounds.size.y;
        curRedState[stateIndex] = mesh.bounds.size.y;
        stateIndex++;
        
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            //Forward Step on Blue Agents + Critics
            NDArray actionDists = blueAgents[i].Forward(curBlueState);
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                if (j == 2) {
                    UnityEngine.Debug.Log(actionDists[j]);
                }
                blueActions[time_step, i, j] = GaussianDistribution.NextGaussian(actionDists[j], blueAgents[i].log_std[j]);
                blueLog_Probs[time_step, i, j] = GaussianDistribution.log_prob(blueActions[time_step, i, j], actionDists[j], blueAgents[i].log_std[j]);
            }
            blueValues[time_step, i] = (blueCritics[i].Forward(curBlueState))[0];

            //Forward Step on Red Agents + Critics
            actionDists = redAgents[i].Forward(curRedState);
            for (int j = 0; j < GameManager.NUM_ACTIONS; j++) {
                redActions[time_step, i, j] = GaussianDistribution.NextGaussian(actionDists[j], redAgents[i].log_std[j]);
                redLog_Probs[time_step, i, j] = GaussianDistribution.log_prob(redActions[time_step, i, j], actionDists[j], redAgents[i].log_std[j]);
            }
            redValues[time_step, i] = (redCritics[i].Forward(curRedState))[0];
        }

        //Apply Forces on Agents
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            //Blue Team Actions
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            Vector3d force = Vector3d.Normalize(new Vector3d(blueActions[time_step, i, 0], 0, blueActions[time_step, i, 1]));
            force *= GameManager.MAX_SPEED*Sigmoid(blueActions[time_step, i, 2]);
            UnityEngine.Debug.Log(force);
            rb.AddForce(new Vector3((float)force[0], (float)force[1], (float)force[2]));

            //Red Team Actions
            rb = redPlayers[i].GetComponent<Rigidbody>();
            force = Vector3d.Normalize(new Vector3d(redActions[time_step, i, 0], 0, redActions[time_step, i, 1]));
            force *= GameManager.MAX_SPEED*Sigmoid(redActions[time_step, i, 2]);
            UnityEngine.Debug.Log(force);
            rb.AddForce(new Vector3((float)force[0], (float)force[1], (float)force[2]));
        }

        time_step++;
    }

    public void getNextValues(NeuralNetwork[] blueCritics, NeuralNetwork[] redCritics) {
        //Common State information for both blue and red
        NDArray curBlueState = new NDArray(GameManager.STATE_SIZE);
        NDArray curRedState = new NDArray(GameManager.STATE_SIZE);
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
        for (int i = 0; i  < GameManager.TEAM_SIZE; i++) {
            blueNextVals[i] = (blueCritics[i].Forward(curBlueState))[0];
            redNextVals[i] = (redCritics[i].Forward(curRedState))[0];
        }
    }



    public void CalculateGAE() {
        for (int i = 0; i < GameManager.TEAM_SIZE; i++) {
            //Gather Blue and Red Agent i's rewards, values, next vals
            NativeArray<double> nativeBlueRewards = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            NativeArray<double> nativeBlueValues = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            double nextBlueVal = blueNextVals[i];
            NativeArray<double> nativeRedRewards = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            NativeArray<double> nativeRedValues = new NativeArray<double>(GameManager.EPISODE_LENGTH, Allocator.TempJob);
            double nextRedVal = redNextVals[i];
            for (int j = 0; j < GameManager.EPISODE_LENGTH; j++) {
                nativeBlueRewards[j] = blueRewards[j, i];
                nativeBlueValues[j] = blueValues[j, i];
                nativeRedRewards[j] = redRewards[j, i];
                nativeRedValues[j] = redValues[j, i];
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
        blueNextVals.Dispose();
        blueReturns.Dispose();
        blueAdvantages.Dispose();
        redNextVals.Dispose();
        redReturns.Dispose();
        redAdvantages.Dispose();
    }

    [BurstCompile]
    private double Sigmoid(double x) {
        return 1 / (math.exp(-x) + 1);
    }
}
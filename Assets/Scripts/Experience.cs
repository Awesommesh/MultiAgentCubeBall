using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using UnityEngine;
using Unity.Burst;
public class Experience {
    //Blue Team
    GameObject[] bluePlayers;
    public GameObject blueGoal;
    public bool blueWon;

    //Red Team   
    GameObject[] redPlayers;
    public GameObject redGoal;
    public bool redWon;

    //General
    GameObject ball;
    public NDArray mask;
    public NativeArray<NDArray> states;
    public NativeArray<NDArray> actions;
    public NativeArray<NDArray> log_probs;
    public NDArray values;
    public NDArray rewards;
    int time_step = 0;
    
    public Experience (GameObject ball, GameObject blueGoal, GameObject redGoal, GameObject[] blueTeam, GameObject[] redTeam) {
        time_step = 0;
        this.ball = ball;
        //General Initialization
        int NUM_TRANSITIONS = GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE * 2;
        NativeArray<int> tempShape = new NativeArray<int>(1, Allocator.Persistent);
        tempShape[0] = GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE * 2;
        mask = NDArray.NDArrayZeros(tempShape, Allocator.Persistent);
        states = new NativeArray<NDArray>(NUM_TRANSITIONS, Allocator.Persistent);
        actions = new NativeArray<NDArray>(NUM_TRANSITIONS, Allocator.Persistent);
        log_probs = new NativeArray<NDArray>(NUM_TRANSITIONS, Allocator.Persistent);
        tempShape = new NativeArray<int>(1, Allocator.Persistent);
        tempShape[0] = GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE * 2;
        values = NDArray.NDArrayZeros(tempShape, Allocator.Persistent);
        tempShape = new NativeArray<int>(1, Allocator.Persistent);
        tempShape[0] = GameManager.EPISODE_LENGTH * GameManager.TEAM_SIZE * 2;
        rewards = NDArray.NDArrayZeros(tempShape, Allocator.Persistent);

        //Blue initialization
        bluePlayers = blueTeam;
        this.blueGoal = blueGoal;
        blueWon = false;

        //Red Initialization
        redPlayers = redTeam;
        this.redGoal = redGoal;
        redWon = false;
    }

    public void stepForward(NeuralNetwork[] blueAgents, NeuralNetwork[] blueCritics, NeuralNetwork[] redAgents, NeuralNetwork[] redCritics) {
        NativeArray<int> stateShape = new NativeArray<int>(1, Allocator.Persistent);
        stateShape[0] = GameManager.STATE_SIZE;
        NDArray blueState = NDArray.NDArrayZeros(stateShape, Allocator.Persistent);
        int stateIndex = 0;
        Rigidbody rb;
        Mesh mesh;
        //Blue State Setup
        for (int i = 0; i < bluePlayers.Length; i++) {
            blueState.set(stateIndex, bluePlayers[i].transform.position.x);
            stateIndex++;
            blueState.set(stateIndex, bluePlayers[i].transform.position.y);
            stateIndex++;
            blueState.set(stateIndex, bluePlayers[i].transform.position.z);
            stateIndex++;
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            blueState.set(stateIndex, rb.velocity.x);
            stateIndex++;
            blueState.set(stateIndex, rb.velocity.y);
            stateIndex++;
            blueState.set(stateIndex, rb.velocity.z);
            stateIndex++;
        }
        for (int i = 0; i < redPlayers.Length; i++) {
            blueState.set(stateIndex, redPlayers[i].transform.position.x);
            stateIndex++;
            blueState.set(stateIndex, redPlayers[i].transform.position.y);
            stateIndex++;
            blueState.set(stateIndex, redPlayers[i].transform.position.z);
            stateIndex++;
            rb = redPlayers[i].GetComponent<Rigidbody>();
            blueState.set(stateIndex, rb.velocity.x);
            stateIndex++;
            blueState.set(stateIndex, rb.velocity.y);
            stateIndex++;
            blueState.set(stateIndex, rb.velocity.z);
            stateIndex++;
        }
        blueState.set(stateIndex, ball.transform.position.x);
        stateIndex++;
        blueState.set(stateIndex, ball.transform.position.y);
        stateIndex++;
        blueState.set(stateIndex, ball.transform.position.z);
        stateIndex++;
        rb = ball.GetComponent<Rigidbody>();
        blueState.set(stateIndex, rb.velocity.x);
        stateIndex++;
        blueState.set(stateIndex, rb.velocity.y);
        stateIndex++;
        blueState.set(stateIndex, rb.velocity.z);
        stateIndex++;

        //Friendly goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<Mesh>();
        blueState.set(stateIndex, blueGoal.transform.position.x);
        stateIndex++;
        blueState.set(stateIndex, blueGoal.transform.position.y);
        stateIndex++;
        blueState.set(stateIndex, blueGoal.transform.position.z);
        stateIndex++;
        blueState.set(stateIndex, mesh.bounds.size.z);
        stateIndex++;
        blueState.set(stateIndex, mesh.bounds.size.y);
        stateIndex++;

        //Enemy goal (!!! z is width and y is height !!!)
        mesh = redGoal.GetComponent<Mesh>();
        blueState.set(stateIndex, redGoal.transform.position.x);
        stateIndex++;
        blueState.set(stateIndex, redGoal.transform.position.y);
        stateIndex++;
        blueState.set(stateIndex, redGoal.transform.position.z);
        stateIndex++;
        blueState.set(stateIndex, mesh.bounds.size.z);
        stateIndex++;
        blueState.set(stateIndex, mesh.bounds.size.y);
        stateIndex++;

        //Red State Setup
        NDArray redState = NDArray.Copy(blueState);
        stateIndex -= 10;
        //Friendly goal (!!! z is width and y is height !!!)
        redState.set(stateIndex, redGoal.transform.position.x);
        stateIndex++;
        redState.set(stateIndex, redGoal.transform.position.y);
        stateIndex++;
        redState.set(stateIndex, redGoal.transform.position.z);
        stateIndex++;
        redState.set(stateIndex, mesh.bounds.size.z);
        stateIndex++;
        redState.set(stateIndex, mesh.bounds.size.y);
        stateIndex++;
        
        //Enemy goal (!!! z is width and y is height !!!)
        mesh = blueGoal.GetComponent<Mesh>();
        redState.set(stateIndex, blueGoal.transform.position.x);
        stateIndex++;
        redState.set(stateIndex, blueGoal.transform.position.y);
        stateIndex++;
        redState.set(stateIndex, blueGoal.transform.position.z);
        stateIndex++;
        redState.set(stateIndex, mesh.bounds.size.z);
        stateIndex++;
        redState.set(stateIndex, mesh.bounds.size.y);
        stateIndex++;

        //All agents and critics Forward
        NativeList<JobHandle>jobList = new NativeList<JobHandle>(Allocator.Temp);

        //Schedule Blue Team Jobs
        for (int i = 0; i < blueAgents.Length; i++) {
            NativeArray<int> actionShape = new NativeArray<int>(1, Allocator.Persistent);
            actionShape[0] = GameManager.NUM_OUTPUTS;
            actions[time_step*GameManager.TEAM_SIZE*2+i] = NDArray.NDArrayZeros(actionShape, Allocator.Persistent);
            NativeArray<int> log_prob_Shape = new NativeArray<int>(1, Allocator.Persistent);
            log_prob_Shape[0] = GameManager.NUM_OUTPUTS;
            log_probs[time_step*GameManager.TEAM_SIZE*2+i] = NDArray.NDArrayZeros(log_prob_Shape, Allocator.Persistent);
            ActorForwardModelJob actorJob = new ActorForwardModelJob {
                model = blueAgents[i],
                state = blueState,
                action = actions[time_step*GameManager.TEAM_SIZE*2+i],
                log_prob = log_probs[time_step*GameManager.TEAM_SIZE*2+i],
            };
            jobList.Add(actorJob.Schedule());
            CriticForwardModelJob criticJob = new CriticForwardModelJob {
                model = blueCritics[i],
                state = blueState,
                value = values[time_step*GameManager.TEAM_SIZE*2+i],
            };
            jobList.Add(criticJob.Schedule());
        }

        //Schedule Red Team Jobs
        int bL = blueAgents.Length;
        for (int i = 0; i < redAgents.Length; i++) {
            NativeArray<int> actionShape = new NativeArray<int>(1, Allocator.Persistent);
            actionShape[0] = GameManager.NUM_OUTPUTS;
            actions[time_step*GameManager.TEAM_SIZE*2+i+bL] = NDArray.NDArrayZeros(actionShape, Allocator.Persistent);
            NativeArray<int> log_prob_Shape = new NativeArray<int>(1, Allocator.Persistent);
            log_prob_Shape[0] = GameManager.NUM_OUTPUTS;
            log_probs[time_step*GameManager.TEAM_SIZE*2+i+bL] = NDArray.NDArrayZeros(log_prob_Shape, Allocator.Persistent);
            ActorForwardModelJob actorJob = new ActorForwardModelJob {
                model = redAgents[i],
                state = redState,
                action = actions[time_step*GameManager.TEAM_SIZE*2+i+bL],
                log_prob = log_probs[time_step*GameManager.TEAM_SIZE*2+i+bL]
            };
            jobList.Add(actorJob.Schedule());
            CriticForwardModelJob criticJob = new CriticForwardModelJob {
                model = redCritics[i],
                state = redState,
                value = values[time_step*GameManager.TEAM_SIZE*2+i+bL]
            };
            jobList.Add(criticJob.Schedule());
        }
        JobHandle.CompleteAll(jobList);

        //Apply Forces on Agents
        //Blue Team Actions
        for (int i = 0; i < bluePlayers.Length; i++) {
            rb = bluePlayers[i].GetComponent<Rigidbody>();
            Vector3d force = Vector3d.Normalize(new Vector3d(actions[time_step*GameManager.TEAM_SIZE*2+i][0], 0, actions[time_step*GameManager.TEAM_SIZE*2+i][1]));
            force *= GameManager.MAX_SPEED*Sigmoid(actions[time_step*GameManager.TEAM_SIZE*2+i][2]);
            rb.AddForce(new Vector3((float)force[0], (float)force[1], (float)force[2]));
        }
        //Red Team Actions
        for (int i = 0; i < redPlayers.Length; i++) {
            rb = redPlayers[i].GetComponent<Rigidbody>();
            Vector3d force = Vector3d.Normalize(new Vector3d(actions[time_step*GameManager.TEAM_SIZE*2+i+bL][0], 0, actions[time_step*GameManager.TEAM_SIZE*2+i+bL][1]));
            force *= GameManager.MAX_SPEED*Sigmoid(actions[time_step*GameManager.TEAM_SIZE*2+i+bL][2]);
            rb.AddForce(new Vector3((float)force[0], (float)force[1], (float)force[2]));
        }
        time_step++;
    }
    
    public double blueReward() {
        if (!redWon && !blueWon) {
            return math.abs(ball.transform.position.x-redGoal.transform.position.x)/GameManager.FIELD_LENGTH;
        } else if (redWon) {
            return -1000;
        } else {
            return 1000;
        }
    }

    public double redReward() {
        if (!redWon && !blueWon) {
            return math.abs(ball.transform.position.x-blueGoal.transform.position.x)/GameManager.FIELD_LENGTH;
        } else if (blueWon) {
            return -1000;
        } else {
            return 1000;
        }
    }

    public void Dispose() {

    }

    [BurstCompile]
    private double Sigmoid(double x) {
        return math.exp(x) / (math.exp(x) + 1);
    }
}
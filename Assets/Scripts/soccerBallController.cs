using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class soccerBallController : MonoBehaviour
{
    public gameManager manager;

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("blueGoal")) //ball touched blue goal
        {
            manager.scoreGoal(true); //blue scored --> handle in game manager
        }
        if (col.gameObject.CompareTag("redGoal")) //ball touched red goal
        {
            manager.scoreGoal(false); //red scored --> handle in game manager
        }
    }
}

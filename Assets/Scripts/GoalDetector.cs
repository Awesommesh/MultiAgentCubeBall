using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GoalDetector : MonoBehaviour
{
	private Transform tf;
	public bool redWon = false;
	public bool blueWon = false;
	public void init() {
		tf = GetComponent<Transform>();
		redWon = false;
		blueWon = false;
	}
    public void checkGoalScored()
    {
    	// Make offset for z in grid if needed
        if (tf.localPosition.x <= -15 && tf.localPosition.z >= -4 && tf.localPosition.z <= 4) {
        	//Blue scored
        	blueWon = true;
        	redWon = false;

		}
		if (tf.localPosition.x >= 15 && tf.localPosition.z >= -4 && tf.localPosition.z <= 4) {
			// Red scored
        	blueWon = false;
			redWon = true;
		}

    }
}

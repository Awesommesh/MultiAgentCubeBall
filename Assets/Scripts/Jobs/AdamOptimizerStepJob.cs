using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;

[BurstCompile]
public struct AdamOptimizerStepJob : IJobParallelFor {
    [ReadOnly]
    public NativeArray<double> weightsGrad;
    [ReadOnly]
    public double alpha;
    [ReadOnly]
    public double beta1;
    [ReadOnly]
    public double beta2;
    [ReadOnly]
    public double epsilon;
    [ReadOnly]
    public int iteration;
    
    public NativeArray<double> V_dw;
    public NativeArray<double> S_dw;
    public NativeArray<double> weights;
    
    public void Execute(int i) {
        V_dw[i] = (beta1 * V_dw[i]) + ((1-beta1) * weightsGrad[i]);
        S_dw[i] = (beta2 * S_dw[i]) + ((1-beta2) * math.pow(weightsGrad[i], 2));
        double V_dw_corrected = V_dw[i] / (1 - math.pow(beta1, iteration)+epsilon);
        double S_dw_corrected = S_dw[i] / (1 - math.pow(beta2, iteration)+epsilon);
        weights[i] -= alpha * (V_dw_corrected / (math.sqrt(S_dw_corrected) + epsilon));
    }
}
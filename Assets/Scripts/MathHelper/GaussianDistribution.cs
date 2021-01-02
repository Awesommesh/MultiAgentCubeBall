using Unity.Mathematics;
using Unity.Collections;
using Unity.Entities;
using Unity.Burst;
using System;

[BurstCompile]
public struct GaussianDistribution : IComponentData {
    public static Unity.Mathematics.Random sampler = new Unity.Mathematics.Random(GameManager.SEED);
    //Mean = 0, STD = 1;
    public static double NextGaussian() {
        /*double v1, v2, s;
        do {
            v1 = 2 * sampler.NextDouble() - 1;
            v2 = 2 * sampler.NextDouble() - 1;
            s = v1 * v2 + v2 * v2;
        } while (s >= 1 || s == 0);
        s = math.sqrt(-2*math.log(s)/s);
        return v1 * s;*/
        double u1 = 1 - sampler.NextDouble();
        double u2 = 1 - sampler.NextDouble();
        return math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.PI * u2);
    }

    public static double NextGaussian(double mean, double std) {
        return mean + NextGaussian() * std;
    }

    public static double NextGaussian(double mean, double std, double range) {
        double nextDouble = NextGaussian(mean, std);
        while(nextDouble > mean + range*std || nextDouble < mean-std*range) {
            nextDouble = NextGaussian(mean, std);
        }
        return nextDouble;
    }

    public static double log_prob(double x, double mean, double std) {
        double var = math.pow(std, 2);
        return -(math.pow(x-mean, 2)/(2*var) - math.log(std) - math.log(math.sqrt(2*math.PI)));
    }

    public static NDArray log_prob(NDArray x, NDArray mean, NDArray std) {
        NDArray log_probs = NDArray.Copy(x);
        for (int i = 0; i < x.numElements; i++) {
            log_probs[i] = log_prob(x[i], mean[i], std[i]);
        }
        return log_probs;
    }

    public static NativeArray<double> log_prob(NativeArray<double> x, NativeArray<double> mean, NativeArray<double> std, Allocator allocator) {
        NativeArray<double> log_probs = new NativeArray<double> (x.Length, allocator);
        for (int i = 0; i < x.Length; i++) {
            log_probs[i] = log_prob(x[i], mean[i], std[i]);
        }
        return log_probs;
    }
    public static NativeArray<double> log_prob(NativeArray<double> x, NativeArray<double> mean, NativeArray<double> std, 
        int startInd, int len, Allocator allocator) {
        NativeArray<double> log_probs = new NativeArray<double> (len, allocator);
        for (int i = startInd; i < startInd+len; i++) {
            log_probs[i-startInd] = log_prob(x[i], mean[i], std[i-startInd]);
        }
        return log_probs;
    }

    //Backwards of log_prob = grad * ((x_t-u(s_t))/std^2)
    public static NDArray log_prob_back(NDArray x, NDArray mean, NDArray std) {
        return (x-mean)/NDArray.Pow(std, 2);
    }

    public static NativeArray<double> log_prob_back(NativeArray<double> x, NativeArray<double> mean, NativeArray<double> std, Allocator allocator) {
        NativeArray<double> log_prob_back = new NativeArray<double> (x.Length, allocator);
        for (int i = 0; i < x.Length; i++) {
            log_prob_back[i] = (x[i]-mean[i])/math.pow(std[i], 2);
        }
        return log_prob_back;
    }

    public static NativeArray<double> log_prob_back(NativeArray<double> x, NativeArray<double> mean, NativeArray<double> std, 
        int startInd, int len, Allocator allocator) {
        NativeArray<double> log_prob_back = new NativeArray<double> (len, allocator);
        for (int i = startInd; i < startInd+len; i++) {
            log_prob_back[i-startInd] = (x[i]-mean[i])/math.pow(std[i-startInd], 2);
        }
        return log_prob_back;
    }

    public static double entropy(double std) {
        return 0.5 + 0.5 * math.log(2 * math.PI) + math.log(std);
    }
}

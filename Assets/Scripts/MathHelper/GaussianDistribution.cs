using Unity.Mathematics;
using Unity.Collections;

public struct GaussianDistribution {
    public static Random sampler = new Random(GameManager.SEED);
    //Mean = 0, STD = 1;
    public static double NextGaussian() {
        double v1, v2, s;
        do {
            v1 = 2 * sampler.NextDouble() - 1;
            v2 = 2 * sampler.NextDouble() - 1;
            s = v1 * v2 + v2 * v2;
        } while (s >= 1 || s == 0);
        s = math.sqrt(-2*math.log(s)/s);
        return v1 * s;
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

    public static NDArray log_prob(NDArray x, NDArray mean, NDArray std, Allocator allocator) {
        NDArray log_probs = NDArray.Copy(x, allocator);
        for (int i = 0; i < x.numElements; i++) {
            log_probs[i] = log_prob(x[i], mean[i], std[i]);
        }
        return log_probs;
    }

    //Backwards of log_prob = grad * ((x_t-u(s_t))/std^2)
    public static NDArray log_prob_back(NDArray x, NDArray mean, NDArray std) {
        return (x-mean)/NDArray.Pow(std, 2);
    }

    public static double entropy(double std) {
        return 0.5 + 0.5 * math.log(2 * math.PI) + math.log(std);
    }
}

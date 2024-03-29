using Unity.Mathematics;

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
}

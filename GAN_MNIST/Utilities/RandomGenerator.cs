using System;

namespace GAN_MNIST
{
    public abstract class RandomGenerator
    {
        public static RandomGenerator Basic { get; } = new PseudoRandom();

        public abstract double GetUniformDouble();

        public abstract double GetUniformDouble(double min, double max);

        public abstract int GetUniformInt32();

        public abstract int GetUniformInt32(int min, int max);

        public virtual double GetNormalDouble()
        {
            double u1 = GetUniformDouble();
            double u2 = GetUniformDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            return r * Math.Sin(theta);
        }

        public virtual double GetNormalDouble(double mean, double standardDeviation)
        {
            if (standardDeviation <= 0.0)
            {
                throw new ArgumentOutOfRangeException($"{nameof(standardDeviation)} must be positive.");
            }
            return mean + standardDeviation * GetNormalDouble();
        }
    }

    public class PseudoRandom : RandomGenerator
    {
        private readonly Random rng;

        public PseudoRandom()
        {
            rng = new Random();
        }

        public PseudoRandom(int seed)
        {
            rng = new Random(seed);
        }

        public override double GetUniformDouble()
        {
            return rng.NextDouble();
        }

        public override double GetUniformDouble(double min, double max)
        {
            return min + GetUniformDouble() * (max - min);
        }

        public override int GetUniformInt32()
        {
            return rng.Next();
        }

        public override int GetUniformInt32(int min, int max)
        {
            return rng.Next(min, max);
        }
    }
}

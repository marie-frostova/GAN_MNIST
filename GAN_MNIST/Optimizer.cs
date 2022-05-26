using System;
using System.Collections.Generic;
using System.Linq;

namespace GAN_MNIST
{
    public abstract class Optimizer
    {
        public Optimizer(string name, double learningRate)
        {
            Name = name;
            LearningRate = learningRate;
        }

        public string Name { get; set; }

        public double LearningRate { get; protected set; }

        public abstract void Update(Layer layer);
    }

    public class AdamOptimizer : Optimizer
    {
        private Dictionary<string, Array> ms;
        private Dictionary<string, Array> vs;
        private long iteration;

        public AdamOptimizer(double learningRate = 0.01, double beta1 = 0.9, double beta2 = 0.999) : base("adam", learningRate)
        {
            Beta1 = beta1;
            Beta2 = beta2;
            ms = new Dictionary<string, Array>();
            vs = new Dictionary<string, Array>();
            iteration = 0;
        }

        public double Beta1 { get; }

        public double Beta2 { get; }

        public override void Update(Layer layer)
        {
            iteration++;
            foreach (var p in layer.Parameters.ToList())
            {
                string paramFullName = $"{layer.Name}_{layer.Guid}_{p.Key}";

                Array weights = p.Value;

                Array grad = layer.Gradients[p.Key];

                if (!ms.ContainsKey(paramFullName))
                {
                    ms[paramFullName] = Array.Zeroes(weights.Rows, weights.Columns);
                    vs[paramFullName] = Array.Zeroes(weights.Rows, weights.Columns);
                }

                ms[paramFullName] = Array.ApplyElementwiseFunction(ms[paramFullName], grad, (m, g) => Beta1 * m + (1 - Beta1) * g);

                vs[paramFullName] = Array.ApplyElementwiseFunction(vs[paramFullName], grad, (v, g) => Beta2 * v + (1 - Beta2) * Math.Pow(g, 2));

                double learningRateForThisIteration = LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, iteration)) / (1 - Math.Pow(Beta1, iteration));
                
                layer.Parameters[p.Key] = Array.ApplyElementwiseFunction(weights, ms[paramFullName], vs[paramFullName], (w, m, v) => w - (learningRateForThisIteration * m / (Math.Sqrt(v) + Network.Epsilon)));
            }
        }
    }
}

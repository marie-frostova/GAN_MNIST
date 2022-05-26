using System;

namespace GAN_MNIST
{
    public abstract class Loss
    {
        public Loss(string name)
        {
            Name = name;
        }

        public string Name { get; set; }

        public abstract Array Forward(Array predictions, Array labels);

        public abstract Array Backward(Array predictions, Array labels);
    }

    public class CrossEntropy : Loss
    {
        public CrossEntropy(string name = "binary cross entropy") : base(name) { }

        public override Array Forward(Array predictions, Array labels)
        {
            var min = Network.Epsilon;
            var max = 1 - Network.Epsilon;
            var clipped = Array.ApplyElementwiseFunction(predictions, p => p < min ? min : p > max ? max : p);
            var output = Array.ApplyElementwiseFunction(clipped, labels, (c, l) => (-(l * Math.Log(c) + (1 - l) * Math.Log(1 - c)))).
            Average();
            return output;
        }

        public override Array Backward(Array predictions, Array labels)
        {
            return Array.ApplyElementwiseFunction(predictions, labels, (p, l) => (p - l) / (p * (1 - p)));
        }
    }
}

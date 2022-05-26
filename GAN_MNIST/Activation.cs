using System;

namespace GAN_MNIST
{
    public abstract class Activation : Layer
    {
        public Activation(string name) : base(name) { }
    }

    public class LeakyReLU : Activation
    {
        private double alpha;

        public LeakyReLU(string name) : base(name)
        {
            alpha = 0d;
        }

        public LeakyReLU(double alpha, string name = "relu") : base($"{name} ({alpha})")
        {
            this.alpha = alpha;
        }

        public override void Forward(Array input)
        {
            base.Forward(input);
            Output = Array.ApplyElementwiseFunction(input, i => i > 0 ? i : i * alpha);
        }

        public override void Backward(Array gradient)
        {
            InputGradient = Array.ApplyElementwiseFunction(gradient, Output!, (g, o) => g * (o > 0 ? 1 : alpha));
        }
    }

    public class Sigmoid : Activation
    {
        public Sigmoid(string name = "sigmoid") : base(name) { }

        public override void Forward(Array input)
        {
            base.Forward(input);
            var exp = Array.ApplyElementwiseFunction(input,
                i => Math.Exp(i)
            );

            Output = Array.ApplyElementwiseFunction(exp,
                ex =>
                ex / (1 + ex)
            );
        }

        public override void Backward(Array gradient)
        {
            InputGradient = Array.ApplyElementwiseFunction(gradient, Output!, (g, o) => g * o * (1 - o));
        }
    }

    public class Tanh : Activation
    {
        public Tanh(string name = "tanh") : base(name) { }

        public override void Forward(Array input)
        {
            base.Forward(input);
            Output = Array.ApplyElementwiseFunction(input, i => Math.Tanh(i));
        }

        public override void Backward(Array gradient)
        {
            InputGradient = Array.ApplyElementwiseFunction(gradient, Output!, (g, o) => g * (1 - Math.Pow(o, 2)));
        }
    }
}

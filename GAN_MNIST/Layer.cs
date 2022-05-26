using System;
using System.Collections.Generic;

namespace GAN_MNIST
{
    public abstract class Layer
    {
        public Layer(string name)
        {
            Parameters = new Dictionary<string, Array>();
            Gradients = new Dictionary<string, Array>();
            Name = name;
        }

        public string Name { get; }

        public Guid Guid { get; } = Guid.NewGuid();

        public Array? Input { get; set; }

        public Array? Output { get; set; }

        public Dictionary<string, Array> Parameters { get; }

        public Array? InputGradient { get; set; }

        public Dictionary<string, Array> Gradients { get; }

        public virtual void Forward(Array input)
        {
            Input = input;
        }

        public abstract void Backward(Array gradient);
    }

    public class DenseLayer : Layer
    {
        public int InputDim { get; }

        public int OutputNeurons { get; }

        public Activation? Activation { get; }

        public DenseLayer(int inputDim, int outputNeurons, Activation? act, string name = "dense") : base(name)
        {
            double range = Math.Sqrt(6d / (inputDim + outputNeurons));
            Parameters["weights"] = Array.UniformRandomised(-range, range, inputDim, outputNeurons);

            InputDim = inputDim;
            OutputNeurons = outputNeurons;
            Activation = act;
        }

        public override void Forward(Array input)
        {
            base.Forward(input);

            Output = input.Multiply(Parameters["weights"]);
            if (Activation != null)
            {
                Activation.Forward(Output);
                Output = Activation.Output;
            }
        }

        public override void Backward(Array gradient)
        {
            if (Activation != null)
            {
                Activation.Backward(gradient);
                gradient = Activation.InputGradient!;
            }
            InputGradient = gradient.Multiply(Parameters["weights"].Transpose());
            Gradients["weights"] = Input!.Transpose().Multiply(gradient);
        }
    }
}

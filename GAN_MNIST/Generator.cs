using System.IO;

namespace GAN_MNIST
{
    public class Generator
    {
        Network network;
        int NoiseDim;

        public Generator(Optimizer optimizer, int noiseDim)
        {
            NoiseDim = noiseDim;
            network = new Network(optimizer, new CrossEntropy());
            network.Add(new DenseLayer(NoiseDim, 256, new LeakyReLU(0.2)));
            network.Add(new DenseLayer(256, 512, new LeakyReLU(0.2)));
            network.Add(new DenseLayer(512, 1024, new LeakyReLU(0.2)));
            network.Add(new DenseLayer(1024, 28 * 28, new Tanh()));
        }

        public double Train(Discriminator discriminator, int epoch, int step, int batchSize)
        {
            var noise = CreateNoise(batchSize);
            var forgedLabels = Array.Ones(batchSize, 1);
            var generatorForward = network.Forward(noise);
            var generatorDiscriminatorForward = discriminator.network.Forward(generatorForward);
            var generatorDiscriminatorLoss = discriminator.network.Cost.Forward(generatorDiscriminatorForward, forgedLabels);
            discriminator.network.Backward(discriminator.network.Cost.Backward(generatorDiscriminatorForward, forgedLabels));
            network.Backward(discriminator.network.Layers[0].InputGradient!);
            foreach (var layer in network.Layers)
            {
                network.Optimizer.Update(layer);
            }
            return generatorDiscriminatorLoss[0];
        }

        public (Array images, Array labels) GetFakeImages(int numberOfImages)
        {
            var noise = CreateNoise(numberOfImages);
            var images = network.Predict(noise);
            var labels = Array.Zeroes(numberOfImages, 1);
            return (images, labels);
        }

        private Array CreateNoise(int numberOfImages)
        {
            return Array.NormalRandomised(0, 1, numberOfImages, NoiseDim);
        }
    }
}

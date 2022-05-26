using System.IO;

namespace GAN_MNIST
{
    public class Discriminator
    {
        public Network network;
        public Discriminator(Optimizer optimizer)
        {
            network = new Network(optimizer, new CrossEntropy());
            network.Add(new DenseLayer(28 * 28, 512, new LeakyReLU(0.2)));
            network.Add(new DenseLayer(512, 256, new LeakyReLU(0.2)));
            network.Add(new DenseLayer(256, 1, new Sigmoid()));
        }

        public double Train(Array XTrain, int halfBatch, Generator generator, int epoch, int step)
        {
            var (realImages, realLabels) = GetRealImages(XTrain, halfBatch);
            var forwardReal = network.Forward(realImages);
            var realLoss = network.Cost.Forward(forwardReal, realLabels);
            network.Backward(network.Cost.Backward(forwardReal, realLabels));
            foreach (var layer in network.Layers)
            {
                network.Optimizer.Update(layer);
            }
            var (fakeImages, fakeLabels) = generator.GetFakeImages(halfBatch);
            var forwardFake = network.Forward(fakeImages);
            var fakeLoss = network.Cost.Forward(forwardFake, fakeLabels);
            network.Backward(network.Cost.Backward(forwardFake, fakeLabels));
            foreach (var layer in network.Layers)
            {
                network.Optimizer.Update(layer);
            }
            var meanLoss = (realLoss[0] + fakeLoss[0]) / 2d;
            return meanLoss;
        }

        private static (Array images, Array labels) GetRealImages(Array XTrain, int numberOfImages)
        {
            int startIdx = RandomGenerator.Basic.GetUniformInt32(0, XTrain.Rows - numberOfImages);
            var images = XTrain.SliceRows(startIdx, numberOfImages);
            var labels = Array.FillWith(0.9, numberOfImages, 1);
            return (images, labels);
        }
    }
}

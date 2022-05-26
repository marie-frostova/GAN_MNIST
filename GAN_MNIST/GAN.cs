using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GAN_MNIST
{
    public class GAN
    {
        int NumEpochs;
        int BatchSize;
        int NoiseDim;
        Array InputMatrix;
        double testProportion = 0.3;

        public GAN(int numEpochs, int batchSize, int noiseDim)
        {
            NumEpochs = numEpochs;
            BatchSize = batchSize;
            NoiseDim = noiseDim;
        }

        public void Run(int number, Array inputMatrix, string imageDirectory)
        {
            InputMatrix = inputMatrix;

            var testCount = (int)(inputMatrix.Rows * testProportion);
            var test = inputMatrix.SliceRows(0, testCount);
            var train = inputMatrix.SliceRows(test.Rows, inputMatrix.Rows - testCount);

            var adam = new AdamOptimizer(learningRate: 2e-4, beta1: 0.5, beta2: 0.999);

            var generator = new Generator(adam, NoiseDim);
            var discriminator = new Discriminator(adam);

            var fileWriter = new StreamWriter(imageDirectory + $"\\error_{number}.txt");
            fileWriter.AutoFlush = true;
            fileWriter.Write($"NumberToTrain: {number}\n");
            fileWriter.Write($"ImagesCount: {train.Rows}\n");
            fileWriter.Write($"EpochsCount: {NumEpochs}\n");

            Train(imageDirectory, number, train, generator, discriminator, fileWriter);
        }

        public void Train(string imageFolder, int number, Array train, Generator generator, Discriminator discriminator, StreamWriter fileWriter)
        {
            int batchCount = (int)Math.Ceiling(train.Rows / (double)BatchSize);
            int halfBatch = (int)Math.Floor(BatchSize / 2d);

            fileWriter.Write($"StepsCount: {batchCount}\n");

            var lossDEpochAvg = new List<double>();
            var lossGEpochAvg = new List<double>();

            for (var epoch = 1; epoch <= NumEpochs; epoch++)
            {
                var lossD = new List<double>();
                var lossG = new List<double>();

                ConsoleExtension.WriteLine($"Number: {number} Epoch: {epoch}");
                train.ShuffleRows();

                for (var step = 1; step <= batchCount; step++)
                {
                    lossD.Add(discriminator.Train(train, halfBatch, generator, epoch, step));
                    lossG.Add(generator.Train(discriminator, epoch, step, BatchSize));
                }

                lossDEpochAvg.Add(lossD.Average());
                lossGEpochAvg.Add(lossG.Average());

                ImageHelper.GenerateExampleImages(generator, 5, 5, imageFolder + $"\\Epoch_{epoch}.png");
            }

            for(var i = 0; i < NumEpochs; i++)
            {
                fileWriter.Write($"{lossDEpochAvg[i]} ");
            }
            fileWriter.Write($"\n");
            for (var i = 0; i < NumEpochs; i++)
            {
                fileWriter.Write($"{lossGEpochAvg[i]} ");
            }
            ImageHelper.GenerateGif(NumEpochs, number, imageFolder);
        }
    }
}


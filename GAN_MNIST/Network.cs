using System;
using System.Collections.Generic;
using System.Linq;

namespace GAN_MNIST
{
    public class Network
    {
        public const double Epsilon = 1e-7;

        public Network(Optimizer optimiser, Loss cost)
        {
            Layers = new List<Layer>();
            Optimizer = optimiser;
            Cost = cost;
        }

        public event EventHandler<BatchEndEventArgs>? BatchEnd;

        public event EventHandler<EpochEndEventArgs>? EpochEnd;

        public List<Layer> Layers { get; }

        public Optimizer Optimizer { get; }

        public Loss Cost { get; }

        public List<double> TrainingLoss { get; set; } = new List<double>();

        public void Add(Layer layer)
        {
            Layers.Add(layer);
        }

        public Array Forward(Array input)
        {
            Layer? previousLayer = null;
            foreach (var layer in Layers)
            {
                if (previousLayer == null)
                    layer.Forward(input);
                else
                    layer.Forward(previousLayer.Output!);
                previousLayer = layer;
            }
            return previousLayer!.Output!;
        }

        public void Backward(Array gradOutput)
        {
            var curGradOutput = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; --i)
            {
                var layer = Layers[i];
                layer.Backward(curGradOutput);
                curGradOutput = layer.InputGradient!;
            }
        }

        public Array Predict(Array inputs)
        {
            return Forward(inputs);
        }

        public void Train(Array trainingData, Array labels, int epochs, int batchSize)
        {
            List<double> batchLoss = new List<double>();

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                var epochLossAvg = TrainBatches(trainingData, labels, batchSize, epoch);

                TrainingLoss.Add(epochLossAvg);

                EpochEndEventArgs eventArgs = new EpochEndEventArgs(epoch, epochLossAvg);
                EpochEnd?.Invoke(epoch, eventArgs);
            }
        }

        private double TrainBatches(Array trainingData, Array labels, int batchSize, int epoch)
        {
            int currentIndex = 0;
            int currentBatch = 1;
            List<double> batchLosses = new List<double>(); ;

            while (trainingData.CanSliceRows(currentIndex, batchSize))
            {
                var xtrain = trainingData.SliceRows(currentIndex, batchSize);
                var ytrain = labels.SliceRows(currentIndex, batchSize);
                var ypred = Forward(xtrain);
                var costVal = Cost.Forward(ypred, ytrain);
                batchLosses.Add(costVal.Data[0]);
                var grad = Cost.Backward(ypred, ytrain);
                Backward(grad);
                foreach (var layer in Layers)
                {
                    Optimizer.Update(layer);
                }
                currentIndex = currentIndex + batchSize;
                double batchLossAvg = Math.Round(costVal.Data[0], 3);

                BatchEndEventArgs eventArgs1 = new BatchEndEventArgs(epoch, currentBatch, batchLossAvg);
                BatchEnd?.Invoke(epoch, eventArgs1);
                currentBatch += 1;
            }

            return Math.Round(batchLosses.Average(), 3);
        }
    }

    public class BatchEndEventArgs
    {
        public BatchEndEventArgs(int epoch, int batch, double loss)
        {
            Epoch = epoch;
            Batch = batch;
            Loss = loss;
        }

        public int Epoch { get; }

        public int Batch { get; }

        public double Loss { get; }

        public override string ToString() => $"Epoch {Epoch} Batch {Batch} Loss {Loss}.";
    }

    public class EpochEndEventArgs
    {
        public EpochEndEventArgs(int epoch, double loss)
        {
            Epoch = epoch;
            Loss = loss;
        }

        public int Epoch { get; }

        public double Loss { get; }

        public override string ToString() => $"Epoch {Epoch} Loss {Loss}";
    }
}

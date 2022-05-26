using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GAN_MNIST
{
    /// <summary>
    /// A helper class for loading the MNIST Dataset.
    /// </summary>
    public class MNISTReader
    {
        private string TrainImagesPath;
        private string TrainLabelsPath;
        private string TestImagesPath;
        private string TestLabelsPath;
        private int padding = 0;
        private double minScale;
        private double maxScale;
        public Dictionary<int, Array> data;
        public List<List<double>> list;
        Dictionary<int, int> count;

        public MNISTReader(string directory, int minScale = 0, int maxScale = 1)
        {
            TrainImagesPath = directory + "\\MNIST\\train-images-idx3-ubyte";
            TrainLabelsPath = directory + "\\MNIST\\train-labels-idx1-ubyte";
            TestImagesPath = directory + "\\MNIST\\t10k-images-idx3-ubyte";
            TestLabelsPath = directory + "\\MNIST\\t10k-labels-idx1-ubyte";
            this.minScale = minScale;
            this.maxScale = maxScale;
            data = new Dictionary<int, Array>();
            count = new Dictionary<int, int>();
            list = new List<List<double>>();
            for(var i = 0; i < 10; i++)
            {
                list.Add(new List<double>());
                count.Add(i, 0);
            }
            LoadAllData();
        }

        public void LoadAllData()
        {
            Read(TrainImagesPath, TrainLabelsPath);
            Read(TestImagesPath, TestLabelsPath);
            for (var i = 0; i < 10; i++)
            {
                data.Add(i, Array.FromData(list[i].ToArray(), count[i], 28 * 28));
            }
        }

        public Array GetMnistData(int number)
        {
            return data[number];
        }

        private void Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = ReadBigInt32(images);
            int numberOfImages = ReadBigInt32(images);
            int width = ReadBigInt32(images);
            int height = ReadBigInt32(images);

            int magicLabel = ReadBigInt32(labels);
            int numberOfLabels = ReadBigInt32(labels);

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var pixelsArray = new double[(height + padding) * (width + padding)];

                for (var j = 0; j < height; j++)
                {
                    for (var k = 0; k < width; k++)
                    {
                        pixelsArray[(j + padding / 2) * height + (k + padding / 2)] = bytes[j * height + k];
                    }
                }
                //Scaling
                for (var j = 0; j < bytes.Length; j++)
                {
                    pixelsArray[j] = MinMaxScale(pixelsArray[j], minScale, maxScale);
                }

                var label = labels.ReadByte();
                list[label].AddRange(pixelsArray);
                count[label]++;
            }
        }

        public static double MinMaxScale(double arg, double min, double max)
        {
            var std = arg / 255;
            return std * (max - min) + min;
        }

        public static int ReadBigInt32(BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) System.Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}

using System;
using System.IO;

namespace GAN_MNIST
{
    class Program
    {
        static void Main(string[] args)
        {
            var projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            var folderDirectory = CreateFolder(projectDirectory, "Generated_Images");

            var mnistHelper = new MNISTReader(projectDirectory, minScale: -1, maxScale: 1);

            var gan = new GAN(numEpochs: 20, batchSize: 256, noiseDim: 100);

            var numbersToGenerate = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            foreach (var number in numbersToGenerate)
            {
                var imageDirectory = CreateFolder(folderDirectory, $"Examples_{number}");
                gan.Run(number, mnistHelper.GetMnistData(number), imageDirectory);
            }
        }

        public static string CreateFolder(string path, string name)
        {
            var folder = path + "\\" + name;
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }
            return folder;
        }
    }
}

using ImageMagick;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace GAN_MNIST
{
    public static class ImageHelper
    {
        public static void GenerateExampleImages(Generator generator, int rows, int cols, string filename)
        {
            var (createdImages, _) = generator.GetFakeImages(rows * cols);
            var pad = 1;
            var width = (28 + 2 * pad) * cols;
            var height = (28 + 2 * pad) * rows;
            var data = new double[width, height];
            for (var x = 0; x < cols; x++)
            {
                for (var y = 0; y < rows; y++)
                {
                    int sx = (28 + 2 * pad) * x;
                    int sy = (28 + 2 * pad) * y;
                    var subImage = createdImages.SliceRows(x * rows + y, 1);
                    for (var i = 0; i < (28 + 2 * pad); i++)
                    {
                        for (var j = 0; j < (28 + 2 * pad); j++)
                        {
                            if (i < pad || i >= (28 + pad) || j < pad || j >= (28 + pad))
                            {
                                data[sx + i, sy + j] = 1;
                            }
                            else
                            {
                                var subImageIdx = 28 * (j - pad) + (i - pad);
                                data[sx + i, sy + j] = subImage[subImageIdx];
                            }
                        }
                    }
                }
            }
            GenerateImage(data, filename);
        }

        public static void GenerateImage(double[,] data, string filename)
        {
            var width = data.GetLength(0);
            var height = data.GetLength(1);

            using (var image = new Image<Rgba32>(width, height))
            {
                for (var x = 0; x < width; x++)
                {
                    for (var y = 0; y < height; y++)
                    {
                        var fd = (float)data[x, y];
                        image[x, y] = new Rgba32(fd, fd, fd);

                    }
                }
                image.Save(filename);
            }
        }

        public static void GenerateGif(int numEpochs, int number, string directory)
        {
            using (MagickImageCollection collection = new MagickImageCollection())
            {
                for (var i = 0; i < numEpochs; i++)
                {
                    collection.Add(directory + $"\\Epoch_{i + 1}.png");
                    collection[0].AnimationDelay = 100;
                }

                collection.Write(directory + $"\\Gif_{number}.gif");
            }
        }
    }
}

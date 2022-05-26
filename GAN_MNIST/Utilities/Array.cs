using System;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GAN_MNIST
{
    public class Array
    {
        public Array(int rows, int columns)
        {
            Rows = rows > 0 ? rows : throw new ArgumentOutOfRangeException($"{nameof(rows)} must be +ve.");
            Columns = columns > 0 ? columns : throw new ArgumentOutOfRangeException($"{nameof(columns)} must be +ve.");
            Count = rows * columns;
            Data = new double[Count];
        }

        public int Rows { get; }

        public int Columns { get; }

        public int Count { get; }

        public double[] Data { get; }

        public double this[int index]
        {
            get
            {
                return Data[index];
            }
            set
            {
                Data[index] = value;
            }
        }

        public double this[int row, int column]
        {
            get
            {
                return Data[row * Columns + column];
            }
            set
            {

                Data[row * Columns + column] = value;
            }
        }

        public Array SliceRows(int startRow, int countOfRows)
        {
            var startIdx = Columns * startRow;
            var countOfElements = Columns * countOfRows;

            var slicedData = Data.Skip(startIdx).Take(countOfElements).ToArray();
            return FromData(slicedData, slicedData.Length / Columns, Columns);
        }

        public bool CanSliceRows(int startRow, int countOfRows)
        {
            var startIdx = Columns * startRow;
            var countOfElements = Columns * countOfRows;
            if (startIdx < 0 || countOfElements <= 0 || startIdx + countOfElements > Data.Length)
            {
                return false;
            }

            return true;
        }

        public Array Average()
        {
            return FillWith(Data.Average(), 1, 1);
        }

        public Array Multiply(Array other)
        {
            if (Columns != other.Rows)
            {
                throw new ArgumentException();
            }

            Array result = new Array(Rows, other.Columns);
            Parallel.For(0, Rows, i => {
                for (int j = 0; j < other.Columns; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < Columns; k++)
                    {
                        result[i, j] += this[i, k] * other[k, j];
                    }
                }
            });
            return result;
        }

        public Array Transpose()
        {
            Array result = new Array(Columns, Rows);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[j, i] = this[i, j];
                }
            }
            return result;
        }

        public void ShuffleRows()
        {
            for (int i = Rows - 1; i > 0; i--)
            {
                var swapIndex = RandomGenerator.Basic.GetUniformInt32(0, i + 1);
                if (swapIndex != i)
                {
                    SwapRows(i, swapIndex);
                }

            }
        }

        private void SwapRows(int r1, int r2)
        {
            for (int i = 0; i < Columns; i++)
            {
                var r1iIndex = r1 * Columns + i;
                var r2iIndex = r2 * Columns + i;

                var tmp = Data[r1iIndex];
                Data[r1iIndex] = Data[r2iIndex];
                Data[r2iIndex] = tmp;
            }
        }

        public static Array FillWith(double fillWith, int rows, int columns)
        {
            var result = new Array(rows, columns);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = fillWith;
            }
            return result;
        }

        public static Array Zeroes(int rows, int columns)
        {
            return FillWith(0d, rows, columns);
        }

        public static Array Ones(int rows, int columns)
        {
            return FillWith(1d, rows, columns);
        }

        public static Array UniformRandomised(double min, double max, int rows, int columns)
        {
            if (min >= max) throw new ArgumentException();
            var result = new Array(rows, columns);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = RandomGenerator.Basic.GetUniformDouble(min, max);
            }
            return result;
        }

        public static Array NormalRandomised(double mean, double standardDeviation, int rows, int columns)
        {
            var result = new Array(rows, columns);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = RandomGenerator.Basic.GetNormalDouble(mean, standardDeviation);
            }
            return result;
        }

        public static Array FromData(double[] data, int rows, int columns)
        {
            var result = new Array(rows, columns);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = data[i];
            }
            return result;
        }

        public static Array ApplyElementwiseFunction(Array first, Func<double, double> f)
        {
            Array result = new Array(first.Rows, first.Columns);

            Parallel.For(0, first.Count, i => { result.Data[i] = f(first[i]); });
            return result;
        }

        public static Array ApplyElementwiseFunction(Array first, Array second, Func<double, double, double> f)
        {
            Array result = new Array(first.Rows, first.Columns);

            Parallel.For(0, first.Count, i => { result.Data[i] = f(first[i], second[i]); });
            return result;
        }

        public static Array ApplyElementwiseFunction(Array first, Array second, Array third, Func<double, double, double, double> f)
        {
            Array result = new Array(first.Rows, first.Columns);

            Parallel.For(0, first.Count, i => { result.Data[i] = f(first[i], second[i], third[i]); });
            return result;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;

namespace GAN_MNIST
{
    public static class ConsoleExtension
    {
        private static string DateTimeString => $"{DateTime.Now.ToString("f")} : ";

        public static void WriteLine(string s)
        {
            Console.WriteLine($"{DateTimeString}{s}");
        }
    }
}

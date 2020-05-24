using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using SharpML.Recurrent.Models;
using SharpML.Recurrent.Networks;
using SharpML.Recurrent.Util;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Trainer;
using SharpML.Reccurent.Examples.Data;

namespace SharpML.Reccurent.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            //ExampleDsr.Run(); incomplete
            //Thread t = new Thread(() => {
            //    ExampleText.Run();
            //}, 1000000000);
            //t.Start();
            //ExampleXor.Run();
            Random rnd = new Random();
            while (true)
            {
                NeuralNetwork network = new NeuralNetwork(null);
                if (ask("chcete síť načíst (y) nebo vygenerovat novou (n)?: "))
                {
                    while (true)
                    {
                        try
                        {
                            Console.Write("Zadejte cestu ke složce se uloženou sítí: ");
                            string path = Console.ReadLine();
                            Console.WriteLine("-------------------------LOADING STARTED----------------------------");
                            network = NetworkBuilder.LoadLSTM(path);
                            Console.WriteLine("-------------------------LOADING COMPLEAT---------------------------");
                            break;
                        }
                        catch
                        {
                            Console.WriteLine("Zadali jste neplatnou cestu");
                        }
                    }
                }
                else
                {
                    int hiddenDimension = askInt("Zadejte dimenzi skryté vrstvy: ");
                    int hiddenLayers = askInt("Zadejte počet skrytých vrstev: ");
                    double init = askDouble("Zadejte hodnotu, která určí rozsah vygenerovaných vah: ");
                    network = NetworkBuilder.MakeLstm(FieldLetterTranslator.Letters.Length, hiddenDimension, hiddenLayers, FieldLetterTranslator.Letters.Length, new SigmoidUnit(), init, rnd);
                }
                if (ask("Přejete si síť naučit (y) nebo jen vygenerovat výstup (n)?: "))
                {
                    int epochs = askInt("Zadejte počet epoch, který se má síť naučit: ");
                    double lr = askDouble("Zadejte počáteční hodnotu learning rate: ");
                    double dr = askDouble("Zadejte hodnotu, kterou chcete learing rate po každé epoše násobit: ");
                    while(true)
                    {
                        try
                        {
                            Console.Write("Zadejte cestu k tréninkovým datům: ");
                            string learningPath = Console.ReadLine();
                            Console.Write("Zadejte cestu ke složce, kam chcete ukládat výstupy z učení: ");
                            string savePath = Console.ReadLine();
                            ExampleText.Run(network, learningPath, savePath, epochs, lr, dr, rnd);
                            break;
                        }
                        catch
                        {
                            Console.WriteLine("zadali jste špatnou cestu");
                        }
                    }
                }
                else
                {
                    int numOfChars = askInt("Zadejte počet znaků, které chcete vygenerovat: ");
                    while (true)
                    {
                        Console.Write("Zadejte cestu k souboru, kam chcete výstup uložit: ");
                        string path = Console.ReadLine();
                        char start = (char)rnd.Next(97, 123);
                        Console.WriteLine("-------------------------GENERATE STARTED----------------");
                        string output = start + ExampleText.generateOutput(network, start, numOfChars);
                        try
                        {
                            using(System.IO.StreamWriter sw = new System.IO.StreamWriter(path))
                            {
                                sw.WriteLine(output);
                            }
                            break;
                        }
                        catch
                        {
                            Console.WriteLine("Zadali jste neplatnou cestu");
                        }                        
                    }
                    Console.WriteLine("-------------------------GENERATE COMPLEAT---------------");
                }
                if (ask("Přejete si aplikaci ukočit?: ")) break;
            }
        }
        static bool ask(string question)
        {
            while(true)
            {
                Console.Write(question);
                string answer = Console.ReadLine().ToUpper();
                if (answer == "YES" || answer == "Y" || answer == "A" || answer == "ANO") return true;
                if (answer == "NO" || answer == "NE" || answer == "N") return false;
                Console.Write("Špatná opověď, ");
            }
        }
        static double askDouble(string question)
        {
            while (true)
            {
                Console.Write(question);
                string answer = Console.ReadLine();
                double d;
                if (double.TryParse(answer, out d)) return d;
                Console.Write("Opověď nemá správný formát, ");
            }
        }
        static int askInt(string question)
        {
            while (true)
            {
                Console.Write(question);
                string answer = Console.ReadLine();
                int i;
                if (int.TryParse(answer, out i)) return i;
                Console.Write("Opověď nemá správný formát, ");
            }
        }
    }
}

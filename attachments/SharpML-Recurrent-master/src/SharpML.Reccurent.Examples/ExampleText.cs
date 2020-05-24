using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using SharpML.Recurrent.DataStructs;
using SharpML.Reccurent.Examples.Data;
using SharpML.Recurrent.Networks;
using SharpML.Recurrent.Util;
using SharpML.Recurrent.Trainer;
using SharpML.Recurrent.Models;
using SharpML.Recurrent.Activations;
using SharpML.Reccurent.Examples.Data;

namespace SharpML.Reccurent.Examples
{
    class ExampleText
    {
        public static void GenerateResults(string path)
        {
            NeuralNetwork network = NetworkBuilder.LoadLSTM(path);
            string output = generateOutput(network, 'a', 1000);
            using (StreamWriter sw = new StreamWriter(Path.Combine(path, "output.txt")))
            {
                sw.WriteLine(output);
            }
        }

        public static void Run()
        {
            Random rnd = new Random();
            FieldLetterTranslator.addChar((char)10);
            Console.WriteLine("Generate started");
            DataSet data = new TextDataSetGenerator(@"C:\Users\Kubik.HOME-PC\Dropbox\shakespear.txt");
            Console.WriteLine("Generate comlpeat");
            int inputDimension = FieldLetterTranslator.Letters.Length;
            int hiddenDimension = 128;
            int outputDimension = FieldLetterTranslator.Letters.Length;
            int hiddenLayers = 2;
            double learningRate = 0.0012;
            double initPatStdDev = 0.08;
            INonlinearity lin = new SigmoidUnit();
            NeuralNetwork network = NetworkBuilder.MakeLstm(inputDimension, hiddenDimension, hiddenLayers, outputDimension, lin, initPatStdDev, rnd);

            string output;


            int reportEveryNthEpoch = 50; 
            int trainingEpochs = 50;

            for (int i = 0; i < trainingEpochs; i++ )
            {
                
                Trainer.train<NeuralNetwork>(1, learningRate, network, data, reportEveryNthEpoch, rnd);
                if (Directory.Exists(@"C:\Users\Kubik.HOME-PC\Documents\NeuralsTraing4\step" + i.ToString()))
                {
                    Directory.Delete(@"C:\Users\Kubik.HOME-PC\Documents\NeuralsTraing4\step" + i.ToString(), true);
                }
                learningRate *= 0.85;
                Directory.CreateDirectory(@"C:\Users\Kubik.HOME-PC\Documents\NeuralsTraing4\step" + i.ToString());
                NetworkBuilder.SaveLSTM(network, @"C:\Users\Kubik.HOME-PC\Documents\NeuralsTraing4\step" + i.ToString());
                output = generateOutput(network, 'a', 1000);
                using (StreamWriter sw = new StreamWriter(Path.Combine(@"C:\Users\Kubik.HOME-PC\Documents\NeuralsTraing4\outputs", "output" + i.ToString() + ".txt")))
                {
                    sw.WriteLine(output);
                }
            }

        }
        public static void Run(NeuralNetwork network, string learningPath, string savePath, int trainingEpochs, double learningRate, double descRate, Random rnd)
        {
            Console.WriteLine("-------------------DATASET GENERATE STARTED--------------------");
            DataSet data = new TextDataSetGenerator(learningPath);                                   
            Console.WriteLine("-------------------DATASET GENERATE COMPLEAT-------------------");

            string output;


            for (int i = 0; i < trainingEpochs; i++)
            {
                Console.WriteLine((i + 1).ToString() + ". EPOCHA");
                Trainer.train<NeuralNetwork>(1, learningRate, network, data, 100000, rnd);
                if (Directory.Exists(savePath + @"\step" + i.ToString()))
                {
                    Directory.Delete(savePath + @"\step" + i.ToString(), true);
                }
                learningRate *= descRate;
                Directory.CreateDirectory(savePath + @"\step" + i.ToString());
                NetworkBuilder.SaveLSTM(network, savePath + @"\step" + i.ToString());
                output = generateOutput(network, 'a', 1000);
                if (!Directory.Exists(Path.Combine(savePath + @"\outputs")))
                {
                    Directory.CreateDirectory(Path.Combine(savePath + @"\outputs"));
                }
                using (StreamWriter sw = new StreamWriter(Path.Combine(savePath + @"\outputs", "output" + i.ToString() + ".txt")))
                {
                    sw.WriteLine(output);
                }
            }

        }
        public static string generateOutput(NeuralNetwork network, char start, int length)
        {
            Matrix input = new Matrix(FieldLetterTranslator.traslateToField(start));
            Graph g = new Graph(false);
            Random rnd = new Random();
            string result = "";
            for(int i = 0;i <length ; i++)
            {
                Matrix output = network.Activate(input, g);
                //char act = FieldLetterTranslator.traslateToLetter(output.W);
                char act = 'a';
                try
                {
                    act = FieldLetterTranslator.Letters[Util.PickIndexFromRandomVector(output, rnd)];
                }
                catch(Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
                input = new Matrix(FieldLetterTranslator.traslateToField(act));
                g = new Graph(false);
                result += act;
            }
            return result.Replace('$', '\n');
        }
    }
}

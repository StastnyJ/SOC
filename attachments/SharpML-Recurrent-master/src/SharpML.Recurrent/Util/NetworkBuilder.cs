using System;
using System.Collections.Generic;
using System.Xml.Serialization;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Networks;
using System.IO;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Util
{    public static class NetworkBuilder
    {

        public static NeuralNetwork MakeLstm(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new LstmLayer(inputDimension, hiddenDimension, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new LstmLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeLstmWithInputBottleneck(int inputDimension, int bottleneckDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            layers.Add(new LinearLayer(inputDimension, bottleneckDimension, initParamsStdDev, rng));
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new LstmLayer(bottleneckDimension, hiddenDimension, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new LstmLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeFeedForward(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity hiddenUnit, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new FeedForwardLayer(inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new FeedForwardLayer(hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeGru(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new GruLayer(inputDimension, hiddenDimension, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new GruLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeRnn(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity hiddenUnit, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new RnnLayer(inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new RnnLayer(hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static void SaveLSTM(NeuralNetwork network, string dir, string dirName = "LSTMNetwork")
        {
            List<ILayer> layers = network._layers;
            //creating main dir
            string actDirectiry = Path.Combine(dir,dirName);
            if(Directory.Exists(actDirectiry))
            {
                Directory.Delete(actDirectiry, true);
            }
            Directory.CreateDirectory(actDirectiry);
            //creating infoFile
            using(StreamWriter sw = new StreamWriter(Path.Combine(actDirectiry, "info.csv")))
            {
                sw.WriteLine(layers.Count - 1);
            }
            //saving output layer
            string outputDirLayer = Path.Combine(actDirectiry, "output");
            Directory.CreateDirectory(outputDirLayer);
            saveFFLayer(Path.Combine(outputDirLayer, "FFLayer"), (FeedForwardLayer)layers.Last());
            //saving hidden layers
            string hiddenDirLayer = Path.Combine(actDirectiry, "hidden");
            Directory.CreateDirectory(hiddenDirLayer);
            for(int i = 0; i < layers.Count - 1; i++)
            {
                saveLSTMLayer(Path.Combine(hiddenDirLayer, "LSTMLayer" + i.ToString()), (LstmLayer)layers[i]);
            }
        }
        public static NeuralNetwork LoadLSTM(string dir, string dirName = "LSTMNetwork")
        {
            
                List<ILayer> layers = new List<ILayer>();
                string actDirectory = Path.Combine(dir, dirName);
                int numberOfHiddenLayers = 0;
                using(StreamReader sr = new StreamReader(Path.Combine(actDirectory, "info.csv")))
                {
                    numberOfHiddenLayers = int.Parse(sr.ReadLine());                       
                }
                //loading hiddenLayers
                string hiddenDirectory = Path.Combine(actDirectory, "hidden");
                for(int i = 0; i < numberOfHiddenLayers; i++)
                {
                    layers.Add(LoadLSTMLayer(Path.Combine(hiddenDirectory, "LSTMLayer" + i.ToString())));
                }
                //loading outputLayer
                string outpuDir = Path.Combine(actDirectory, "output");
                layers.Add(loadFFLayer(Path.Combine(outpuDir, "FFLayer")));
                return new NeuralNetwork(layers);
           
            return null;
        }

        static void saveLSTMLayer(string dir, LstmLayer layer)
        {
            Directory.CreateDirectory(dir);
            //make infoFile
            using(StreamWriter sw = new StreamWriter(Path.Combine(dir, "info.csv")))
            {
                sw.WriteLine(layer._inputDimension);
                sw.WriteLine(layer._outputDimension);
            }
            //saving matrixes
            saveMatrix(Path.Combine(dir, "wix.csv"), layer._wix);
            saveMatrix(Path.Combine(dir, "wih.csv"), layer._wih);
            saveMatrix(Path.Combine(dir, "inputBias.csv"), layer._inputBias);
            saveMatrix(Path.Combine(dir, "wfx.csv"), layer._wfx);
            saveMatrix(Path.Combine(dir, "wfh.csv"), layer._wfh);
            saveMatrix(Path.Combine(dir, "forgetBias.csv"), layer._forgetBias);
            saveMatrix(Path.Combine(dir, "wox.csv"), layer._wox);
            saveMatrix(Path.Combine(dir, "woh.csv"), layer._woh);
            saveMatrix(Path.Combine(dir, "outputBias.csv"), layer._outputBias);
            saveMatrix(Path.Combine(dir, "wcx.csv"), layer._wcx);
            saveMatrix(Path.Combine(dir, "wch.csv"), layer._wch);
            saveMatrix(Path.Combine(dir, "cellWriteBias.csv"), layer._cellWriteBias);
            saveMatrix(Path.Combine(dir, "hiddenContext.csv"), layer._hiddenContext);
            saveMatrix(Path.Combine(dir, "cellContext.csv"), layer._cellContext);
        }
        static LstmLayer LoadLSTMLayer(string dir)
        {
            try
            {
                int inputDimension;
                int outputDimension;

                using(StreamReader sr = new StreamReader(Path.Combine(dir,"info.csv")))
                {
                    inputDimension = int.Parse(sr.ReadLine());
                    outputDimension = int.Parse(sr.ReadLine());
                }
                Matrix wix = loadMatrix(Path.Combine(dir, "wix.csv"));
                Matrix wih = loadMatrix(Path.Combine(dir, "wih.csv"));
                Matrix inputBias = loadMatrix(Path.Combine(dir, "inputBias.csv"));
                Matrix wfx = loadMatrix(Path.Combine(dir, "wfx.csv"));
                Matrix wfh = loadMatrix(Path.Combine(dir, "wfh.csv"));
                Matrix forgetBias = loadMatrix(Path.Combine(dir, "forgetBias.csv"));
                Matrix wox = loadMatrix(Path.Combine(dir, "wox.csv"));
                Matrix woh = loadMatrix(Path.Combine(dir, "woh.csv"));
                Matrix outputBias = loadMatrix(Path.Combine(dir, "outputBias.csv"));
                Matrix wcx = loadMatrix(Path.Combine(dir, "wcx.csv"));
                Matrix wch = loadMatrix(Path.Combine(dir, "wch.csv"));
                Matrix cellWriteBias = loadMatrix(Path.Combine(dir, "cellWriteBias.csv"));

                Matrix hiddenContext = loadMatrix(Path.Combine(dir, "hiddenContext.csv"));
                Matrix cellContext = loadMatrix(Path.Combine(dir, "cellContext.csv"));
                return new LstmLayer(inputDimension, outputDimension, wix, wih, inputBias, wfx, wfh, forgetBias, wox, woh, outputBias, wcx, wch, cellWriteBias, hiddenContext, cellContext);

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return null;
        }
        static void saveFFLayer(string dir, FeedForwardLayer layer)
        {
            Directory.CreateDirectory(dir);
            //make infoFile
            string linearityType = "";
            if (layer._f is LinearUnit) linearityType = "LinearUnit";
            if (layer._f is RectifiedLinearUnit) linearityType = "RectifiedLinearUnit";
            if (layer._f is SigmoidUnit) linearityType = "SigmoidUnit";
            if (layer._f is SineUnit) linearityType = "SineUnit";
            if (layer._f is TanhUnit) linearityType = "TanhUnit";
            using (StreamWriter sw = new StreamWriter(Path.Combine(dir, "info.csv")))
            {
                sw.WriteLine(linearityType);
            }
            //saving matrixes
            saveMatrix(Path.Combine(dir, "W.csv"), layer._w);
            saveMatrix(Path.Combine(dir, "B.csv"), layer._b);
        }
        static FeedForwardLayer loadFFLayer(string dir)
        {
            try
            { 
                //loading noLinearity
                string noLinearity = "";
                using(StreamReader sr = new StreamReader(Path.Combine(dir, "info.csv")))
                {
                    noLinearity = sr.ReadLine();
                }
                INonlinearity lin = null;
                if(noLinearity == "LinearUnit") lin = new LinearUnit();
                if (noLinearity == "RectifiedLinearUnit") lin = new RectifiedLinearUnit();
                if (noLinearity == "SigmoidUnit") lin = new SigmoidUnit();
                if (noLinearity == "SineUnit") lin = new SineUnit();
                if (noLinearity == "TanhUnit") lin = new TanhUnit();
                Matrix w = loadMatrix(Path.Combine(dir, "W.csv"));
                Matrix b = loadMatrix(Path.Combine(dir, "B.csv"));
                return new FeedForwardLayer(w, b, lin);
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return null;
        }
        static void saveMatrix(string file, Matrix matrix)
        {
            using(StreamWriter sw = new StreamWriter(file))
            {
                if(matrix == null)
                {
                    sw.WriteLine("null");
                    return;
                }
                sw.WriteLine("notNull");
                sw.WriteLine(matrix.Rows);
                sw.WriteLine(matrix.Cols);
                sw.WriteLine(string.Join(";", (IEnumerable<double>)(matrix.W.ToList())));
                sw.WriteLine(string.Join(";", (IEnumerable<double>)(matrix.Dw.ToList())));
                sw.WriteLine(string.Join(";", (IEnumerable<double>)(matrix.StepCache.ToList())));
            }
        }
        static Matrix loadMatrix(string file)
        {
            try
            {
                int rows;
                int cols;
                double[] w;
                double[] dw;
                double[] stepCache;
                using(StreamReader sr = new StreamReader(file))
                {
                    if (sr.ReadLine() == "null") return null;
                    rows = int.Parse(sr.ReadLine());
                    cols = int.Parse(sr.ReadLine());
                    //loading w field
                    string[] input = sr.ReadLine().Split(';');
                    w = new double[input.Length];
                    for(int i = 0; i < w.Length; i++)
                    {
                        w[i] = double.Parse(input[i]);
                    }
                    //loading dw field
                    input = sr.ReadLine().Split(';');
                    dw = new double[input.Length];
                    for (int i = 0; i < dw.Length; i++)
                    {
                        dw[i] = double.Parse(input[i]);
                    }
                    //loading stepCache field
                    input = sr.ReadLine().Split(';');
                    stepCache = new double[input.Length];
                    for (int i = 0; i < stepCache.Length; i++)
                    {
                        stepCache[i] = double.Parse(input[i]);
                    }
                }
                return new Matrix(rows, cols, w, dw, stepCache);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return null;
        }

    }
}

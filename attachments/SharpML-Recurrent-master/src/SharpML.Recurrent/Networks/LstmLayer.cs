using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    [Serializable]
    public class LstmLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
        public int _inputDimension;
        public readonly int _outputDimension;

        public readonly Matrix _wix;
        public readonly Matrix _wih;
        public readonly Matrix _inputBias;
        public readonly Matrix _wfx;
        public readonly Matrix _wfh;
        public readonly Matrix _forgetBias;
        public readonly Matrix _wox;
        public readonly Matrix _woh;
        public readonly Matrix _outputBias;
        public readonly Matrix _wcx;
        public readonly Matrix _wch;
        public readonly Matrix _cellWriteBias;

        public Matrix _hiddenContext;
        public Matrix _cellContext;

        public readonly INonlinearity _inputGateActivation = new SigmoidUnit();
        public readonly INonlinearity _forgetGateActivation = new SigmoidUnit();
        public readonly INonlinearity _outputGateActivation = new SigmoidUnit();
        public readonly INonlinearity _cellInputActivation = new TanhUnit();
        public readonly INonlinearity _cellOutputActivation = new TanhUnit();

        public LstmLayer(int inputDimension, int outputDinmension, Matrix wix, Matrix wih, Matrix inputBias, Matrix wfx, Matrix wfh, Matrix forgetBias, Matrix wox, 
            Matrix woh, Matrix outputBias, Matrix wcx, Matrix wch, Matrix cellWriteBias, Matrix hiddenContext, Matrix cellContext)
        {
            _inputDimension = inputDimension;
            _outputDimension = outputDinmension;
            _wix = wix;
            _wih = wih;
            _inputBias = inputBias;
            _wfx = wfx;
            _wfh = wfh;
            _forgetBias = forgetBias;
            _wox = wox;
            _woh = woh;
            _outputBias = outputBias;
            _wcx = wcx;
            _wch = wch;
            _cellWriteBias = cellWriteBias;
            _hiddenContext = hiddenContext;
            _cellContext = cellContext;
        }

        public LstmLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng)
        {
            this._inputDimension = inputDimension;
            this._outputDimension = outputDimension;
            _wix = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wih = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _inputBias = new Matrix(outputDimension);
            _wfx = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wfh = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            //set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            _forgetBias = Matrix.Ones(outputDimension, 1);
            _wox = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _woh = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _outputBias = new Matrix(outputDimension);
            _wcx = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wch = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _cellWriteBias = new Matrix(outputDimension);
        }

        public Matrix Activate(Matrix input, Graph g)
        {

            //input gate
            Matrix sum0 = g.Mul(_wix, input);
            Matrix sum1 = g.Mul(_wih, _hiddenContext);
            Matrix inputGate = g.Nonlin(_inputGateActivation, g.Add(g.Add(sum0, sum1), _inputBias));

            //forget gate
            Matrix sum2 = g.Mul(_wfx, input);
            Matrix sum3 = g.Mul(_wfh, _hiddenContext);
            Matrix forgetGate = g.Nonlin(_forgetGateActivation, g.Add(g.Add(sum2, sum3), _forgetBias));

            //output gate
            Matrix sum4 = g.Mul(_wox, input);
            Matrix sum5 = g.Mul(_woh, _hiddenContext);
            Matrix outputGate = g.Nonlin(_outputGateActivation, g.Add(g.Add(sum4, sum5), _outputBias));

            //write operation on cells
            Matrix sum6 = g.Mul(_wcx, input);
            Matrix sum7 = g.Mul(_wch, _hiddenContext);
            Matrix cellInput = g.Nonlin(_cellInputActivation, g.Add(g.Add(sum6, sum7), _cellWriteBias));

            //compute new cell activation
            Matrix retainCell = g.Elmul(forgetGate, _cellContext);
            Matrix writeCell = g.Elmul(inputGate, cellInput);
            Matrix cellAct = g.Add(retainCell, writeCell);

            //compute hidden state as gated, saturated cell activations
            Matrix output = g.Elmul(outputGate, g.Nonlin(_cellOutputActivation, cellAct));

            //rollover activations for next iteration
            _hiddenContext = output;
            _cellContext = cellAct;

            return output;
        }

        public void ResetState()
        {
            _hiddenContext = new Matrix(_outputDimension);
            _cellContext = new Matrix(_outputDimension);
        }

        public List<Matrix> GetParameters()
        {
            List<Matrix> result = new List<Matrix>();
            result.Add(_wix);
            result.Add(_wih);
            result.Add(_inputBias);
            result.Add(_wfx);
            result.Add(_wfh);
            result.Add(_forgetBias);
            result.Add(_wox);
            result.Add(_woh);
            result.Add(_outputBias);
            result.Add(_wcx);
            result.Add(_wch);
            result.Add(_cellWriteBias);
            return result;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using SharpML.Recurrent.DataStructs;
using SharpML.Recurrent.Loss;

namespace SharpML.Reccurent.Examples.Data
{
    class TextDataSetGenerator : DataSet
    {
        public TextDataSetGenerator(string trainigFile, string validationFile =  null, string testFile = null)
        {
            InputDimension = FieldLetterTranslator.Letters.Length;
            OutputDimension = FieldLetterTranslator.Letters.Length;
            LossReporting = new LossSoftmax();
            LossTraining = new LossSoftmax();
            Training = getData(trainigFile);
            Validation = (validationFile == null) ? null : (trainigFile == validationFile) ? Training : getData(validationFile);
            Testing = (testFile == null) ? null : (trainigFile == testFile) ? Training : getData(testFile);
        }
        List<DataSequence> getData(string fileName)
        {
            List<DataSequence> result = new List<DataSequence>();
            string row = "";
            using (StreamReader sr = new StreamReader(fileName))
            {
                for (int rowNumber = 0; null != ((row = sr.ReadLine())); rowNumber++ )
                {
                    if (rowNumber % 1000 == 0) Console.WriteLine(rowNumber);
                    DataSequence sequence = new DataSequence();
                    sequence.Steps = new List<DataStep>();
                    for (int i = 0; i < row.Length; i++)
                    {
                        DataStep step = new DataStep(FieldLetterTranslator.traslateToField(row[i]), FieldLetterTranslator.traslateToField((i + 1 < row.Length) ? row[i + 1] : '$'));
                        if (step.Input != null && step.TargetOutput != null) sequence.Steps.Add(step);
                    }
                    result.Add(sequence);
                }
            }
            return result;
        }
    }
    class FieldLetterTranslator
    {
        private const string FIELD_PATH = @"C:\Users\Kubik.HOME-PC\Dropbox\SharpML-Recurrent-master\src\SharpML.Reccurent.Examples\Data/letters.txt";
        private static char[] letters = null;
        public static char[] Letters
        {
            get
            {
                if (letters != null) return letters;
                List<char> res = new List<char>();
                using (StreamReader sr = new StreamReader(FIELD_PATH))
                {
                    string[] input = sr.ReadLine().Split(';');
                    foreach (string act in input)
                    {
                        if (string.IsNullOrEmpty(act)) continue;
                        res.Add((char)int.Parse(act));
                    }
                }
                letters = res.ToArray();
                return letters;
            }
        }
        public static void addChar(char ch)
        {
            if (letters == null) letters = Letters;
            string data;
            using (StreamReader sr = new StreamReader(FIELD_PATH))
            {
                data = sr.ReadLine();
            }
            if (!data.Contains(((int)ch).ToString()))
            {
                List<char> lst = letters.ToList();
                lst.Add(ch);
                letters = lst.ToArray();
                data += ";" + ((int)ch).ToString();
                using(StreamWriter sw = new StreamWriter(FIELD_PATH))
                {
                    sw.WriteLine(data);
                }
            }
        }
        private static int findIndex(char[] field, char letter)
        {
            for (int i = 0; i < field.Length; i++)
            {
                if (field[i] == letter) return i;
            }
            return -1;
        }
        public static double[] traslateToField(char letter)
        {
            if(letter == (char)65533) letter = '.';
            if (letters == null) letters = Letters;
            int index = findIndex(letters, letter.ToString().ToLower().First());
            if (index == -1)
            {
                addChar(letter.ToString().ToLower().First());
                return traslateToField(letter);
            }
            double[] res = new double[letters.Length];
            for (int i = 0; i < res.Length; i++) res[i] = 0;
            res[index] = 1;
            return res;
        }
        
        public static char traslateToLetter(double[] field)
        {
            Random rnd = new Random();
            double max = rnd.NextDouble();
            double sum = 0;
            for (int i = 0; i < field.Length; i++ )
            {
                sum += field[i];
                if (sum >= max) return letters[i];
            }
            throw new ArgumentException();
        }
    }
}
/**
 * Naive Bayes Classifier
 * @author: miRx923
 * @year: 2023
 */


namespace NaiveBayesClassifier
{
    internal class Program
    {
        static void Main(string[] args)
        {
            /**
             * Training with Play-Tennis dataset
             */
            NaiveBayesClassifier classifier1 = new NaiveBayesClassifier();

            using (var reader = new StreamReader("play_tennis-TrainingData.csv"))
            {

                string line = reader.ReadLine(); // skips first line


                while ((line = reader.ReadLine()) != null)
                {
                    string[] values = line.Split(',');
                    string label = values[values.Length - 1]; // saves the class label to variable

                    classifier1.Train(new string[] { values[1], values[2], values[3], values[4] }, label);
                }
            }


            /**
             * Testing with Play-Tennis dataset
             */

            double truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;

            using (var reader = new StreamReader("play_tennis-TestingData.csv"))
            {
                string line = reader.ReadLine(); // skips first line

                while ((line = reader.ReadLine()) != null)
                {
                    string[] values = line.Split(',');
                    string actualLabel = values[values.Length - 1];

                    string[] features = new string[] { values[1], values[2], values[3], values[4] };
                    string predictedLabel = classifier1.Predict(features);

                    if (predictedLabel == "Yes" && actualLabel == "Yes")
                    {
                        truePositive++;
                    }
                    else if (predictedLabel == "Yes" && actualLabel == "No")
                    {
                        falsePositive++;
                    }
                    else if (predictedLabel == "No" && actualLabel == "No")
                    {
                        trueNegative++;
                    }
                    else if (predictedLabel == "No" && actualLabel == "Yes")
                    {
                        falseNegative++;
                    }
                }
            }

            // Efectiveness metrics
            double precision = truePositive / (truePositive + falsePositive);
            double recall = truePositive / (truePositive + falseNegative);
            double f1 = 2 * precision * recall / (precision + recall);
            double accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative);

            // Others efectiveness metrics
            double errorRate = (falsePositive + falseNegative) / (truePositive + trueNegative + falsePositive + falseNegative);
            double TPR = truePositive / (truePositive + falseNegative);
            double TNR = trueNegative / (trueNegative + falsePositive);
            double PPV = truePositive / (truePositive + falsePositive);
            double NPV = trueNegative / (trueNegative + falseNegative);
            double FNR = falseNegative / (falseNegative + truePositive);
            double FPR = falsePositive / (falsePositive + trueNegative);

            Console.WriteLine(  $"Testing with Play-Tennis dataset: \n" +
                                $"Precision: {precision} \n" +
                                $"Recall: {recall} \n" +
                                $"F1: {f1} \n" +
                                $"Accuracy: {accuracy} \n\n" +
                                $"Other efectiveness metrics: \n" +
                                $"ErrorRate: {errorRate}   \n" +
                                $"TruePositiveRate(TPR): {TPR}  \n" +
                                $"TrueNegativeRate(TNR): {TNR}  \n" +
                                $"PositivePredictiveValue(PPV): {PPV}  \n" +
                                $"NegativePredictiveValue(NPV): {NPV}  \n" +
                                $"TruePositiveRate(FNR): {FNR} \n" +
                                $"FalsePositiveRate(FPR): {FPR} \n\n");





            /**
             * Training with Iris-Species dataset
             */
            
            NaiveBayesClassifier classifier2 = new NaiveBayesClassifier();

            using (var reader = new StreamReader("Iris-TrainingData.csv"))
            {
                string line = reader.ReadLine(); // skips first line

                while ((line = reader.ReadLine()) != null)
                {
                    string[] values = line.Split(','); 
                    string label = values[values.Length - 1]; // saves class label to variable

                    classifier2.Train(new string[] { values[1], values[2], values[3], values[4] }, label);
                }
            }
            

            /**
             * Testing with Iris-Species dataset
             */

            
            truePositive = 0;
            trueNegative = 0;
            falsePositive = 0; 
            falseNegative = 0;

            using (var reader = new StreamReader("Iris-TestingData.csv"))
            {
                string line = reader.ReadLine(); // skips first line
                while ((line = reader.ReadLine()) != null)
                {
                    string[] values = line.Split(',');
                    string actualLabel = values[values.Length - 1]; // saves class label to variable

                    string[] features = new string[] { values[1], values[2], values[3], values[4] };
                    string predictedLabel = classifier2.Predict(features);

                    if (predictedLabel == "Iris-setosa" && actualLabel == "Iris-setosa")
                    {
                        truePositive++;
                    }
                    else if (predictedLabel == "Iris-setosa" && actualLabel == "Iris-virginica")
                    {
                        falsePositive++;
                    }
                    else if (predictedLabel == "Iris-virginica" && actualLabel == "Iris-virginica")
                    {
                        trueNegative++;
                    }
                    else if (predictedLabel == "Iris-virginica" && actualLabel == "Iris-setosa")
                    {
                        falseNegative++;
                    }
                }
            }

            
            // Efectiveness metrics
            precision = truePositive / (truePositive + falsePositive);
            recall = truePositive / (truePositive + falseNegative);
            f1 = 2 * precision * recall / (precision + recall);
            accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative);

            // Other efectiveness metrics
            errorRate = (falsePositive + falseNegative) / (truePositive + trueNegative + falsePositive + falseNegative);
            TPR = truePositive / (truePositive + falseNegative);
            TNR = trueNegative / (trueNegative + falsePositive);
            PPV = truePositive / (truePositive + falsePositive);
            NPV = trueNegative / (trueNegative + falseNegative);
            FNR = falseNegative / (falseNegative + truePositive);
            FPR = falsePositive / (falsePositive + trueNegative);


            Console.WriteLine($"Testing with Iris-Species dataset: \n" +
                                $"Precision: {precision} \n" +
                                $"Recall: {recall} \n" +
                                $"F1: {f1} \n" +
                                $"Accuracy: {accuracy} \n\n" +
                                $"Other efectiveness metrics: \n" +
                                $"ErrorRate: {errorRate}   \n" +
                                $"TruePositiveRate(TPR): {TPR}  \n" +
                                $"TrueNegativeRate(TNR): {TNR}  \n" +
                                $"PositivePredictiveValue(PPV): {PPV}  \n" +
                                $"NegativePredictiveValue(NPV): {NPV}  \n" +
                                $"TruePositiveRate(FNR): {FNR} \n" +
                                $"FalsePositiveRate(FPR): {FPR} \n\n");
        }
    }
}

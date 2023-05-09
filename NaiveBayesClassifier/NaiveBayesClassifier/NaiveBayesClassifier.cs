/**
 * Class for implementing Naive Bayes Classifier
 */

namespace NaiveBayesClassifier
{
    
    public class NaiveBayesClassifier
    {
        private Dictionary<string, int> classCounts; // how many times a class is in dataset
        private Dictionary<string, Dictionary<string, int>> featureCounts; // how many times a feature is in a class
        private double numOfExamples; // number of examples in training set
           

        public NaiveBayesClassifier()
        {
            classCounts = new Dictionary<string, int>();
            featureCounts = new Dictionary<string, Dictionary<string, int>>();
            numOfExamples = 0; 
        }


        /// <summary>
        /// Trains the classifier using the features of the example and the class to which the example belongs.
        /// </summary>
        /// <param name="features">An array of strings representing features for training the classifier.</param>
        /// <param name="label">String representing the class to which the example belongs.</param>
        public void Train(string[] features, string label)
        {
            // Catching the errors
            if (string.IsNullOrEmpty(label))
            {
                throw new ArgumentException("Label cannot be null or empty.");
            }

            if (features == null || features.Length == 0)
            {
                throw new ArgumentNullException("Features list cannot be empty.");
            }

            // Training
            if (!classCounts.ContainsKey(label))
            {
                classCounts[label] = 0;
            }

            classCounts[label]++;

            foreach (string feature in features) 
            {
                if (!featureCounts.ContainsKey(label))
                {
                    featureCounts[label] = new Dictionary<string, int>();
                }

                if (!featureCounts[label].ContainsKey(feature))
                {
                    featureCounts[label][feature] = 0;
                }
                featureCounts[label][feature]++;
            }

            numOfExamples++;
        }


        /// <summary>
        /// Predicts the example's class based on the example's properties.
        /// </summary>
        /// <param name="features">Array of strings representing features of the example.</param>
        /// <returns>String class into which the example was classified based on the highest probability.</returns>
        public string Predict(string[] features)
        {
            // Catching the errors
            if (features == null || features.Length == 0)
            {
                throw new ArgumentNullException("Features list cannot be empty.");
            }

            // Predicting
            string bestLabel = "";
            double bestScore = double.MinValue;

            foreach (var _class in classCounts)
            {
                string label = _class.Key;
                double classCountInDataset = _class.Value;

                /*
                 probability of class in a training set:
                 ->    P(trieda) 
                */
                double score = classCountInDataset / numOfExamples;

                foreach (string feature in features)
                {
                    if (featureCounts.ContainsKey(label) && featureCounts[label].ContainsKey(feature))
                    {
                        /*
                         probability of class in a training set sums up with probabilities of features for class:
                         ->    P(trieda) + ∑ P(atribút_i  | trieda) )
                        */
                        score += featureCounts[label][feature] / classCountInDataset;
                    }
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestLabel = label;
                }
            }

            return bestLabel;
        }
    }
}

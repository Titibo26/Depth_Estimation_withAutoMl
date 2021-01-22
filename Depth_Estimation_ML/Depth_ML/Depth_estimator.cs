using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Depth_Estimation_ML.Depth_ML
{
    class Depth_estimator
    {



        // This will get the current WORKING directory (i.e. \bin\Debug)
        static string workingDirectory = Environment.CurrentDirectory;
        // This will get the current PROJECT directory
        static string projDirectory = Directory.GetParent(workingDirectory).Parent.Parent.FullName;
        static readonly string _assetsPath = projDirectory +  "\\Depth_ML";

        static readonly string _imagesFolder = Path.Combine(_assetsPath, "data\\nyu2_train");
        static readonly string _trainTagsCsv = Path.Combine(_assetsPath, "nyu2_subset_train.csv");

        private static IDataView TrainDataView = null;

        public static float[] Mean = { 123.675f, 116.28f, 103.53f };
        public static float[] Scale = { 51.525f, 50.4f, 50.625f };

        public static void GenerateModel()
        {
            var mlContext = new MLContext();


            //My CSV file named "_trainTagsCsv" contains 1 column with the path of a rgb img followed by the path of the corresponding depth img (separated by ',')
            //EX : data/nyu2_train/living_room_0038_out/37.jpg,data/nyu2_train/living_room_0038_out/37.png

            //Feeding the columnInference with the CSV containing  previouse data
            ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(_trainTagsCsv, 0, false, ',', null, null, false, false);
            //columnInference should provide one column with paths to rgb image and one column with paths to depth image
            
            // Load data from files using inferred columns.
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            TrainDataView = textLoader.Load(_trainTagsCsv);

            // STEP 1: Display first few rows of the training data.
            ConsoleHelper.ShowDataViewInConsole(mlContext, TrainDataView);
            DataViewSchema.Column nc = new DataViewSchema.Column();
       

            // STEP 2: Build a pre-featurizer for use in the AutoML experiment.
             IEstimator<ITransformer> preFeaturizer = mlContext.Transforms.Conversion.MapValue("rgb_input",TrainDataView, TrainDataView.Schema[0], TrainDataView.Schema[0], TrainDataView.Schema[0].Name)
               .Append(mlContext.Transforms.Conversion.MapValue("depth_input", TrainDataView, TrainDataView.Schema[1], TrainDataView.Schema[1], TrainDataView.Schema[1].Name));
           
            // STEP 3: Customize column information returned by InferColumns API.
            ColumnInformation columnInformation = columnInference.ColumnInformation;
            columnInformation.CategoricalColumnNames.Remove(TrainDataView.Schema[1].Name);
            columnInformation.CategoricalColumnNames.Remove(TrainDataView.Schema[0].Name);


            columnInformation.ImagePathColumnNames.Add("rgb_input");
            columnInformation.ImagePathColumnNames.Add("depth_input");

            columnInformation.ImagePathColumnNames.Remove("col1");

            columnInformation.LabelColumnName.Remove(0, columnInformation.LabelColumnName.Length);


            columnInformation.IgnoredColumnNames.Add(TrainDataView.Schema[1].Name);
            columnInformation.IgnoredColumnNames.Add(TrainDataView.Schema[0].Name);
     

            IDataView FinalTrainDataView = preFeaturizer.Fit(TrainDataView).Transform(TrainDataView);

            ConsoleHelper.ShowDataViewInConsole(mlContext, FinalTrainDataView);


            // STEP 4: Initialize a cancellation token source to stop the experiment.
            var cts = new CancellationTokenSource();

            // STEP 5: Initialize our user-defined progress handler that AutoML will 
            // invoke after each model it produces and evaluates.
            var progressHandler = new RegressionExperimentProgressHandler();

            // STEP 6: Create experiment settings
            var experimentSettings = new RegressionExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 3600;

            // Set the metric that AutoML will try to optimize over the course of the experiment.
            experimentSettings.OptimizingMetric = RegressionMetric.RootMeanSquaredError;

            // Set the cache directory to null.
            // This will cause all models produced by AutoML to be kept in memory 
            // instead of written to disk after each run, as AutoML is training.
            experimentSettings.CacheDirectory = null;

            // Don't use LbfgsPoissonRegression and OnlineGradientDescent trainers during this experiment.
            experimentSettings.Trainers.Remove(RegressionTrainer.LbfgsPoissonRegression);
            experimentSettings.Trainers.Remove(RegressionTrainer.OnlineGradientDescent);

            // Cancel experiment after the user presses any key
            experimentSettings.CancellationToken = cts.Token;
            CancelExperimentAfterAnyKeyPress(cts);


            // STEP 7: Run AutoML regression experiment.
            var experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);

            ConsoleHelper.ConsoleWriteHeader("=============== Running AutoML experiment ===============");
            Console.WriteLine($"Running AutoML regression experiment...");
            var stopwatch = Stopwatch.StartNew();
            // Cancel experiment after the user presses any key.
            CancelExperimentAfterAnyKeyPress(cts);

                ExperimentResult<RegressionMetrics> experimentResult = experiment.Execute(FinalTrainDataView, columnInformation, preFeaturizer, progressHandler);
                Console.WriteLine($"{experimentResult.RunDetails.Count()} models were returned after {stopwatch.Elapsed.TotalSeconds:0.00} seconds{Environment.NewLine}");

            //ERROR THROWN : System.ArgumentException : 'Provided label column 'Label' was of type String, but only type Single is allowed.'



            /*
            IDataView predictions = model.Transform(TestDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: LabelColumnName, scoreColumnName: "Score");
           */

            //To use if i need to manipulate images ?
            /*
                                          .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: MidasSettings.ImageWidth, imageHeight: MidasSettings.ImageHeight, inputColumnName: "input"))
                                          .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input"))
            */

        }

        public class ImageData
        {
            [LoadColumn(0)]
            public string InputImagePath;

            [LoadColumn(1)]
            public string InputDepthPath; //Tableau des valeurs des pixels ??
        }

        private static void CancelExperimentAfterAnyKeyPress(CancellationTokenSource cts)
        {
            Task.Run(() =>
            {
                Console.WriteLine("Press any key to stop the experiment run...");
                Console.Read();
                cts.Cancel();
            });
        }

        public static IEnumerable<ImageData> ReadFromCsvToImage(string file)
        {

            var tab = File.ReadAllLines(file);

            List<ImageData> myExctractedImages = new List<ImageData>();

            foreach (string s in tab)
            {
                string[] ns = new string[2];
                ns = s.Split(',');
                ImageData newImg = new ImageData();
                newImg.InputImagePath = _assetsPath + ns[0];
                newImg.InputDepthPath = _assetsPath + ns[1];
                myExctractedImages.Add(newImg);
            }


            return myExctractedImages;
        }

    }



   

}

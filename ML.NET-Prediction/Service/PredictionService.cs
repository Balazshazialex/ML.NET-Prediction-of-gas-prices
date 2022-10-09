using Microsoft.ML;
using ML.NET_Prediction.Model;

namespace ML.NET_Prediction.Service
{
    public class PredictionService
    {
        public static List<InputModel> CreateTestData(DateTime? dateTime)
        {
            // Initialize some variables to guide or "random" data generation
            Random rd = new Random();
            int lowEnd = 135;
            int highEnd = 195;

            var inputData = new List<InputModel>();
            // If no default start date is specified, we will generate data from January 1st 2022
            var startDate = dateTime ?? new DateTime(2021, 1, 1);
            // We will generate data up until the current day
            var today = DateTime.Now;
            var index = 1;
            while (startDate <= today)
            {
                // Generate a random price
                var price = rd.Next(lowEnd, highEnd);
                // Add historic data ( each day will be a number, with the first day being day 0, the second one being day 1 and so one )
                inputData.Add(new InputModel((float)index, (float)price));
                index++;
                startDate = startDate.AddDays(1);
            }

            return inputData;
        }

        public static float PredictionWrapper(List<InputModel> dataInitial, float valueToPredict)
        {
            MLContext mlContext = new MLContext(seed: 0);
            var model = Train(mlContext, dataInitial);
            //SaveModel(mlContext, dataInitial, model);
            // If you want to use the already trained model, comment out the train and save part and just uncomment the Load part
            // model=LoadModel(mlContext);
            var prediction = Predict(mlContext, model, valueToPredict);
            return prediction;
        }

        private static ITransformer Train(MLContext mlContext, List<InputModel> dataInitial)
        {
            IDataView dataView = mlContext.Data.LoadFromEnumerable<InputModel>(dataInitial);
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Price")
                                               .Append(mlContext.Transforms.Concatenate("Features", new string[] { "Day" }))
                                               .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                                               .Append(mlContext.Regression.Trainers.Sdca());
            var model = pipeline.Fit(dataView);
            return model;
        }

        private static float Predict(MLContext mlContext, ITransformer model, float valueToPredict)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<InputModel, OutputModel>(model);
            var prediction = predictionFunction.Predict(new InputModel(valueToPredict, 0));
            return prediction.Price;
        }

        public static void SaveModel(MLContext mlContext, List<InputModel> dataInitial, ITransformer model)
        {
            IDataView data = mlContext.Data.LoadFromEnumerable<InputModel>(dataInitial);
            mlContext.Model.Save(model, data.Schema, "model.zip");
        }

        public static ITransformer LoadModel(MLContext mlContext)
        {
            DataViewSchema modelSchema;
            // Load trained model
            ITransformer trainedModel = mlContext.Model.Load("model.zip", out modelSchema);
            return trainedModel;
        }
    }
}

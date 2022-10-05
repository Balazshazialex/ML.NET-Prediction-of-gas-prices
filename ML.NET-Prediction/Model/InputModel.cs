using Microsoft.ML.Data;

namespace ML.NET_Prediction.Model
{
    public class InputModel
    {
        [LoadColumn(0)]
        public float Day;
        [LoadColumn(1)]
        public float Price;
        public InputModel(float day, float price)
        {
            Day = day;
            Price = price;
        }
    }
}

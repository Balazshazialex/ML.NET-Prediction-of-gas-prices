using Microsoft.ML.Data;

namespace ML.NET_Prediction.Model
{
    public class OutputModel
    {
        [ColumnName("Score")]
        public float Price;
    }
}

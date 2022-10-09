using ML.NET_Prediction.Service;
public class Program
{
    public static void Main()
    {
        var list=PredictionService.CreateTestData(null);
        var val=PredictionService.PredictionWrapper(list, 1590f);
    }
}

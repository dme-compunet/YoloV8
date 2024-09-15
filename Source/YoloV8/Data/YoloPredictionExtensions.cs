namespace Compunet.YoloV8.Data;

public static class YoloPredictionExtensions
{
    public static Classification GetTopClass(this YoloResult<Classification> result) => result[0];
}
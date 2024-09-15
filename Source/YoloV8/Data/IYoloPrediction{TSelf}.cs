namespace Compunet.YoloV8.Data;

public interface IYoloPrediction<TSelf>
{
    internal abstract static string Describe(TSelf[] predictions);
}
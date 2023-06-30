namespace Compunet.YoloV8.Data
{
    public interface IKeypoint
    {
        int Index { get; }

        Point Point { get; }

        float Confidence { get; }
    }
}
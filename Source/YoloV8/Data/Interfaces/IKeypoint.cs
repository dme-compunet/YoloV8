namespace Compunet.YoloV8.Data
{
    public interface IKeypoint
    {
        int Id { get; }

        Point Point { get; }

        float Confidence { get; }
    }
}
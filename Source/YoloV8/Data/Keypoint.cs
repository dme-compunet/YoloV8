namespace Compunet.YoloV8.Data;

internal class Keypoint : IKeypoint
{
    public int Id { get; }

    public Point Point { get; }

    public float Confidence { get; }

    public Keypoint(int id, int x, int y, float confidence)
    {
        Id = id;
        Point = new Point(x, y);
        Confidence = confidence;
    }
}
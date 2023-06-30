namespace Compunet.YoloV8.Data;

internal class Keypoint : IKeypoint
{
    public int Index { get; }

    public Point Point { get; }

    public float Confidence { get; }

    public Keypoint(int index, int x, int y, float confidence)
    {
        Index = index;
        Point = new Point(x, y);
        Confidence = confidence;
    }
}
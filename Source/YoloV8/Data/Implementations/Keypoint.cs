namespace Compunet.YoloV8.Data;

public class Keypoint : IKeypoint
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
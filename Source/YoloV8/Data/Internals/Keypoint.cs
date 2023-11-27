namespace Compunet.YoloV8.Data;

internal class Keypoint(int index, int x, int y, float confidence) : IKeypoint
{
    public int Index { get; } = index;

    public Point Point { get; } = new Point(x, y);

    public float Confidence { get; } = confidence;
}
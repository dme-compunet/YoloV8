namespace Compunet.YoloV8.Data;

internal class Mask(float[,] xy) : IMask
{
    private readonly float[,] _xy = xy;

    public float this[int x, int y] => _xy[x, y];

    public int Width { get; } = xy.GetLength(0);

    public int Height { get; } = xy.GetLength(1);

    public float GetConfidence(int x, int y)
    {
        return _xy[x, y];
    }
}
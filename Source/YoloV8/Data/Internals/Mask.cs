namespace Compunet.YoloV8.Data;

internal class Mask : IMask
{
    private readonly float[,] _xy;

    public float this[int x, int y] => _xy[x, y];

    public int Width { get; }

    public int Height { get; }

    public Mask(float[,] xy)
    {
        Width = xy.GetLength(0);
        Height = xy.GetLength(1);

        _xy = xy;
    }
    public float GetConfidence(int x, int y)
    {
        return _xy[x, y];
    }
}
namespace Compunet.YoloV8.Data;

public interface IMask
{
    public float this[int x, int y] { get; }

    public int Width { get; }

    public int Height { get; }

    public float GetConfidence(int x, int y);
}
namespace Compunet.YoloV8.Utilities;

internal readonly struct Vector<T>(T x, T y)
{
    public static Vector<T> Default = new();

    public T X => x;

    public T Y => y;

    public override string ToString() => $"X = {x}, Y = {y}";
}
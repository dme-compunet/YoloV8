namespace Compunet.YoloV8;

public readonly struct Vector<T>(T x, T y)
{
    public static Vector<T> Default { get; } = new();

    public T X => x;

    public T Y => y;

    public override string ToString() => $"X = {x}, Y = {y}";
}
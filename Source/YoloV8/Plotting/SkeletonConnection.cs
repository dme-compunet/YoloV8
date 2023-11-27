namespace Compunet.YoloV8.Plotting;

public readonly struct SkeletonConnection(int first, int second)
{
    public int First { get; } = first;

    public int Second { get; } = second;
}
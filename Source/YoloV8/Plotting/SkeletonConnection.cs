namespace Compunet.YoloV8.Plotting;

public readonly struct SkeletonConnection
{
    public int First { get; }

    public int Second { get; }

    public SkeletonConnection(int first, int second)
    {
        First = first;
        Second = second;
    }
}
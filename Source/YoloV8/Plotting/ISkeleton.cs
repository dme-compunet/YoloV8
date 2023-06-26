namespace Compunet.YoloV8.Plotting;

public interface ISkeleton
{
    IReadOnlyList<SkeletonConnection> Connections { get; }

    Color GetKeypointColor(int index);

    Color GetLineColor(int index);
}
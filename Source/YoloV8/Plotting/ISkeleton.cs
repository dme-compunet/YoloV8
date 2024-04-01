namespace Compunet.YoloV8.Plotting;

public interface ISkeleton
{
    SkeletonConnection[] Connections { get; }

    Color GetKeypointColor(int index);

    Color GetLineColor(int index);
}
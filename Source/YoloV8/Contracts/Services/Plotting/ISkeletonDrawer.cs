namespace Compunet.YoloV8.Contracts.Services.Plotting;

internal interface ISkeletonDrawer
{
    public void DrawSkeleton(Pose prediction, PlottingContext context);
}
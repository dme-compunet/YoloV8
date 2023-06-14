using Compunet.YoloV8.Timing;

namespace Compunet.YoloV8.Data;

internal class PoseResult : YoloV8Result, IPoseResult
{
    public IReadOnlyList<IPose> Poses { get; }

    public PoseResult(Size image,
                      SpeedResult speed,
                      IReadOnlyList<IPose> poses)
        : base(image, speed)
    {
        Poses = poses;
    }

    public override string ToString()
    {
        return $"{Poses.Count} Persons";
    }
}
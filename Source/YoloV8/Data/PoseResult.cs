using Compunet.YoloV8.Timing;

namespace Compunet.YoloV8.Data;

internal class PoseResult : YoloV8Result, IPoseResult
{
    public IReadOnlyList<IPose> Persons { get; }

    public PoseResult(Size image,
                      SpeedResult speed,
                      IReadOnlyList<IPose> poses)
        : base(image, speed)
    {
        Persons = poses;
    }

    public override string ToString()
    {
        return $"{Persons.Count} Persons";
    }
}
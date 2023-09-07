namespace Compunet.YoloV8.Data;

internal class PoseResult : YoloV8Result, IPoseResult
{
    public IReadOnlyList<IPoseBoundingBox> Boxes { get; }

    public PoseResult(Size image,
                      SpeedResult speed,
                      IReadOnlyList<IPoseBoundingBox> boxes)
        : base(image, speed)
    {
        Boxes = boxes;
    }

    public override string ToString()
    {
        return Boxes.Summary();
    }
}
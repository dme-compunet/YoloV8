namespace Compunet.YoloV8.Data;

internal class PoseResult(Size image,
                          SpeedResult speed,
                          IReadOnlyList<IPoseBoundingBox> boxes) : YoloV8Result(image, speed), IPoseResult
{
    public IReadOnlyList<IPoseBoundingBox> Boxes { get; } = boxes;

    public override string ToString()
    {
        return Boxes.Summary();
    }
}
using Compunet.YoloV8.Timing;

namespace Compunet.YoloV8.Data;

internal class DetectionResult : YoloV8Result, IDetectionResult
{
    public IReadOnlyList<IBoundingBox> Boxes { get; init; }

    public string Text => string.Join(", ", Boxes.Select(x => x.ToString()));

    public DetectionResult(Size image,
                           SpeedResult speed,
                           IReadOnlyList<IBoundingBox> boxes)
        : base(image, speed)
    {
        Boxes = boxes;
    }

    public override string ToString()
    {
        var sort = Boxes.Select(x => x.Name)
                        .GroupBy(x => x.Id)
                        .OrderBy(x => x.Key)
                        .Select(x => $"{x.Count()} {x.First().Name}");

        return string.Join(", ", sort);
    }
}
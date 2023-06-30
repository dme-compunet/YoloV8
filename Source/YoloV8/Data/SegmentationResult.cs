using Compunet.YoloV8.Timing;

namespace Compunet.YoloV8.Data;

internal class SegmentationResult : YoloV8Result, ISegmentationResult
{
    public IReadOnlyList<ISegmentationBoundingBox> Boxes { get; }

    public SegmentationResult(Size image, 
                              SpeedResult speed,
                              IReadOnlyList<ISegmentationBoundingBox> boxes) 
        : base(image, speed)
    {
        Boxes = boxes;
    }

    public override string ToString()
    {
        var sort = Boxes.Select(x => x.Class)
                        .GroupBy(x => x.Id)
                        .OrderBy(x => x.Key)
                        .Select(x => $"{x.Count()} {x.First().Name}");

        return string.Join(", ", sort);
    }
}
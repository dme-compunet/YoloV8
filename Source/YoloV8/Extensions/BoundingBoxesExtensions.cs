using Compunet.YoloV8.Data;

namespace Compunet.YoloV8.Extensions;

public static class BoundingBoxesExtensions
{
    public static string Summary(this IReadOnlyList<IBoundingBox> boxes)
    {
        var sort = boxes.Select(x => x.Class)
                        .GroupBy(x => x.Id)
                        .OrderBy(x => x.Key)
                        .Select(x => $"{x.Count()} {x.First().Name}");

        return string.Join(", ", sort);
    }
}
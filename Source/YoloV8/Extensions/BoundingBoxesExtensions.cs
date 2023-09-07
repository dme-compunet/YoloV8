namespace Compunet.YoloV8.Extensions;

public static class BoundingBoxesExtensions
{
    public static string Summary(this IEnumerable<IBoundingBox> boxes)
    {
        var sort = boxes.Select(x => x.Class)
                        .GroupBy(x => x.Id)
                        .OrderBy(x => x.Key)
                        .Select(x => $"{x.Count()} {x.First().Name}");

        return string.Join(", ", sort);
    }
}
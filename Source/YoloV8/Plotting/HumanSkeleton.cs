namespace Compunet.YoloV8.Plotting;

internal class HumanSkeleton : ISkeleton
{
    private readonly string[] _colors = new[]
    {
        "FF8000",
        "FF9933",
        "FFB266",
        "E6E600",
        "FF99FF",
        "99CCFF",
        "FF66FF",
        "FF33FF",
        "66B2FF",
        "3399FF",
        "FF9999",
        "FF6666",
        "FF3333",
        "99FF99",
        "66FF66",
        "33FF33",
        "00FF00",
        "0000FF",
        "FF0000",
        "FFFFFF",
    };

    private readonly SkeletonConnection[] _connections = new SkeletonConnection[]
    {
        new(15, 13),
        new(13, 11),
        new(16, 14),
        new(14, 12),
        new(11, 12),
        new(5, 11),
        new(6, 12),
        new(5, 6),
        new(5, 7),
        new(6, 8),
        new(7, 9),
        new(8, 10),
        new(1, 2),
        new(0, 1),
        new(0, 2),
        new(1, 3),
        new(2, 4),
        new(3, 5),
        new(4, 6)
    };

    private readonly int[] _keypointColorMap = new[]
    {
        16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9
    };

    private readonly int[] _lineColorMap = new[]
    {
        9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16
    };

    public IReadOnlyList<SkeletonConnection> Connections => _connections;

    public Color GetKeypointColor(int index)
    {
        index = _keypointColorMap[index % _keypointColorMap.Length];

        var hex = _colors[index];

        return Color.ParseHex(hex);
    }

    public Color GetLineColor(int index)
    {
        index = _lineColorMap[index % _lineColorMap.Length];

        var hex = _colors[index];

        return Color.ParseHex(hex);
    }
}
namespace Compunet.YoloV8.Plotting;

public class ColorPalette
{
    public static ColorPalette Default { get; } = CreateDefault();

    private readonly Func<int, string> _selector;

    public ColorPalette(string color)
    {
        _selector = _ => color;
    }

    public ColorPalette(string[] colors)
    {
        _selector = index => colors[index % colors.Length];
    }

    public ColorPalette(Func<int, string> selector)
    {
        _selector = selector;
    }

    public Color GetColor(int index)
    {
        var hex = _selector(index);

        return Color.ParseHex(hex);
    }

    private static ColorPalette CreateDefault()
    {
        return new ColorPalette(new string[]
        {
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        });
    }
}
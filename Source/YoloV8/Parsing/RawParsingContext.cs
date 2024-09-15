namespace Compunet.YoloV8.Parsing;

internal readonly ref struct RawParsingContext
{
    public required DenseTensor<float> Tensor { get; init; }

    public required Vector<int> Padding { get; init; }

    public required Vector<float> Ratio { get; init; }

    public required int Stride1 { get; init; }

    public int NameCount { get; init; }
}
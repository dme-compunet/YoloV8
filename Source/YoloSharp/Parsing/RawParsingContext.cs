namespace Compunet.YoloSharp.Parsing;

internal readonly ref struct RawParsingContext
{
    public required YoloArchitecture Architecture { get; init; }

    public required MemoryTensor<float> Tensor { get; init; }

    public int NameCount { get; init; }
}
namespace Compunet.YoloV8.Metadata;

public readonly struct KeypointShape(int count, int channels)
{
    public int Count { get; } = count;

    public int Channels { get; } = channels;
}
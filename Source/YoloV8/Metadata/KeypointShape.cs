namespace Compunet.YoloV8.Metadata;

public readonly struct KeypointShape
{
    public int Count { get; }

    public int Channels { get; }

    public KeypointShape(int count, int channels)
    {
        Count = count;
        Channels = channels;
    }
}
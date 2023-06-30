namespace Compunet.YoloV8.Metadata;

public class YoloV8PoseMetadata : YoloV8Metadata
{
    public KeypointShape KeypointShape { get; }

    public YoloV8PoseMetadata(string author,
                              string description,
                              string version,
                              YoloV8Task task,
                              Size imageSize,
                              IReadOnlyList<YoloV8Class> classes,
                              KeypointShape keypointShape)
        : base(author,
               description,
               version,
               task,
               imageSize,
               classes)
    {
        KeypointShape = keypointShape;
    }
}
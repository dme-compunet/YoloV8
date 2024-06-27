namespace Compunet.YoloV8.Metadata;

public class YoloV8Metadata(string author,
                            string description,
                            string version,
                            YoloV8Task task,
                            int batch,
                            Size imageSize,
                            IReadOnlyList<YoloV8Class> names)
{
    public static YoloV8Metadata Parse(IDictionary<string, string> metadata)
    {
        var author = metadata["author"];
        var description = metadata["description"];
        var version = metadata["version"];

        var task = metadata["task"] switch
        {
            "obb" => YoloV8Task.Obb,
            "pose" => YoloV8Task.Pose,
            "detect" => YoloV8Task.Detect,
            "segment" => YoloV8Task.Segment,
            "classify" => YoloV8Task.Classify,
            _ => throw new InvalidOperationException("Unknow YoloV8 'task' value")
        };

        var batch = int.Parse(metadata["batch"]);

        var imageSize = ParseSize(metadata["imgsz"]);
        var classes = ParseNames(metadata["names"]);

        if (task is YoloV8Task.Pose)
        {
            var keypointShape = ParseKeypointShape(metadata["kpt_shape"]);

            return new YoloV8PoseMetadata(author,
                                          description,
                                          version,
                                          task,
                                          batch,
                                          imageSize,
                                          classes,
                                          keypointShape);
        }

        return new YoloV8Metadata(author,
                                  description,
                                  version,
                                  task,
                                  batch,
                                  imageSize,
                                  classes);
    }

    public string Author { get; } = author;

    public string Description { get; } = description;

    public string Version { get; } = version;

    public YoloV8Task Task { get; } = task;

    public int Batch { get; } = batch;

    public Size ImageSize { get; } = imageSize;

    public IReadOnlyList<YoloV8Class> Names { get; } = names;

    #region Static Parsers

    private static Size ParseSize(string text)
    {
        text = text[1..^1]; // '[640, 641]' => '640, 640'

        var split = text.Split(", ");

        var y = int.Parse(split[0]);
        var x = int.Parse(split[1]);

        return new Size(x, y);
    }

    private static KeypointShape ParseKeypointShape(string text)
    {
        text = text[1..^1]; // '[17, 3]' => '17, 3'

        var split = text.Split(", ");

        var count = int.Parse(split[0]);
        var channels = int.Parse(split[1]);

        return new KeypointShape(count, channels);
    }

    private static YoloV8Class[] ParseNames(string text)
    {
        text = text[1..^1];

        var split = text.Split(", ");
        var count = split.Length;

        var names = new YoloV8Class[count];

        for (int i = 0; i < count; i++)
        {
            var value = split[i];

            var valueSplit = value.Split(": ");

            var id = int.Parse(valueSplit[0]);
            var name = valueSplit[1][1..^1].Replace('_', ' ');

            names[i] = new YoloV8Class(id, name);
        }

        return names;
    }

    #endregion
}
namespace Compunet.YoloV8.Metadata;

public class YoloMetadata
{
    public string Author { get; }

    public string Description { get; }

    public string Version { get; }

    public int BatchSize { get; }

    public Size ImageSize { get; }

    public YoloTask Task { get; }

    public YoloName[] Names { get; }

    public YoloArchitecture Architecture { get; }

    internal YoloMetadata(InferenceSession session)
        : 
        this(session.ModelMetadata.CustomMetadataMap) 
    { }
    internal YoloMetadata(Dictionary<string, string> metadata)
    {
        Author = metadata["author"];
        Description = metadata["description"];
        Version = metadata["version"];

        Task = metadata["task"] switch
        {
            "obb" => YoloTask.Obb,
            "pose" => YoloTask.Pose,
            "detect" => YoloTask.Detect,
            "segment" => YoloTask.Segment,
            "classify" => YoloTask.Classify,
            _ => throw new InvalidOperationException("Unknow YoloV8 'task' value")
        };

        Architecture = GetYoloArchitecture(Description);
        BatchSize = int.Parse(metadata["batch"]);
        ImageSize = ParseSize(metadata["imgsz"]);
        Names = ParseNames(metadata["names"]);
    }

    public static YoloMetadata Parse(InferenceSession session)
    {
        try
        {
            if (session.ModelMetadata.CustomMetadataMap["task"] == "pose")
            {
                return new YoloPoseMetadata(session);
            }

            return new YoloMetadata(session);
        }
        catch (Exception inner)
        {
            throw new InvalidOperationException("The metadata parsing failed, making sure you use an official YOLOv8 model", inner);
        }
    }

    private static YoloArchitecture GetYoloArchitecture(string description)
    {
        if (description.Contains("yolov8", StringComparison.CurrentCultureIgnoreCase))
        {
            return YoloArchitecture.YoloV8;
        }

        if (description.Contains("yolov10", StringComparison.CurrentCultureIgnoreCase))
        {
            return YoloArchitecture.YoloV10;
        }

        if (description.Contains("yolo11", StringComparison.CurrentCultureIgnoreCase))
        {
            return YoloArchitecture.Yolo11;
        }

        throw new NotSupportedException("Unrecognized YOLO model architecture");
    }

    #region Parsers

    private static Size ParseSize(string text)
    {
        text = text[1..^1]; // '[640, 640]' => '640, 640'

        var split = text.Split(", ");

        var y = int.Parse(split[0]);
        var x = int.Parse(split[1]);

        return new Size(x, y);
    }

    private static YoloName[] ParseNames(string text)
    {
        text = text[1..^1];

        var split = text.Split(", ");
        var count = split.Length;

        var names = new YoloName[count];

        for (int i = 0; i < count; i++)
        {
            var value = split[i];

            var valueSplit = value.Split(": ");

            var id = int.Parse(valueSplit[0]);
            var name = valueSplit[1][1..^1].Replace('_', ' ');

            names[i] = new YoloName(id, name);
        }

        return names;
    }

    #endregion
}
namespace Compunet.YoloV8.Metadata;

public class YoloV8Metadata
{
    public static YoloV8Metadata Parse(IDictionary<string, string> metadata)
    {
        var author = metadata["author"];
        var description = metadata["description"];
        var version = metadata["version"];

        var task = metadata["task"] switch
        {
            "detect" => YoloV8Task.Detect,
            "segment" => YoloV8Task.Segment,
            "pose" => YoloV8Task.Pose,
            "classify" => YoloV8Task.Classify,
            _ => throw new ArgumentException("Unknow YoloV8 'task' value")
        };

        var imageSize = ParseSize(metadata["imgsz"]);
        var names = ParseNames(metadata["names"]);

        return new YoloV8Metadata(author,
                                  description,
                                  version,
                                  task,
                                  imageSize,
                                  names);
    }

    public string Author { get; }

    public string Description { get; }

    public string Version { get; }

    public YoloV8Task Task { get; }

    public Size ImageSize { get; }

    public YoloV8Class[] Names { get; }

    public YoloV8Metadata(string author,
                          string description,
                          string version,
                          YoloV8Task task,
                          Size imageSize,
                          YoloV8Class[] names)
    {
        Author = author;
        Description = description;
        Version = version;
        Task = task;
        ImageSize = imageSize;
        Names = names;
    }

    private static Size ParseSize(string text)
    {
        text = text[1..^1]; // '[640, 641]' => '640, 640'

        var split = text.Split(", ");

        var x = int.Parse(split[0]);
        var y = int.Parse(split[1]);

        return new Size(x, y);
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
            var name = valueSplit[1][1..^1];

            names[i] = new YoloV8Class(id, name);
        }

        return names;
    }
}
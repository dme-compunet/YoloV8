namespace Compunet.YoloV8;

public class YoloV8Configuration
{
    public static readonly YoloV8Configuration Default = new();

    public float Confidence { get; set; }

    public float IoU { get; set; }

    public bool KeepOriginalAspectRatio { get; set; }

    public bool SuppressParallelInference { get; set; }

    public YoloV8Configuration()
    {
        Confidence = .3f;
        IoU = .45f;
        KeepOriginalAspectRatio = true;
        SuppressParallelInference = false;
    }
}
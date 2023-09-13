namespace Compunet.YoloV8;

public class YoloV8Parameters
{
    public static readonly YoloV8Parameters Default = new();

    public float Confidence { get; set; }

    public float IoU { get; set; }

    public bool ProcessWithOriginalAspectRatio { get; set; }

    public YoloV8Parameters()
    {
        Confidence = .3f;
        IoU = .45f;
        ProcessWithOriginalAspectRatio = false;
    }
}
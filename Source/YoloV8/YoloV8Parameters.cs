namespace Compunet.YoloV8;

public class YoloV8Parameters
{
    public static readonly YoloV8Parameters Default = new(confidence: .3f,
                                                          iou: .45f);

    public float Confidence { get; set; }

    public float IoU { get; set; }

    public YoloV8Parameters(float confidence, float iou)
    {
        Confidence = confidence;
        IoU = iou;
    }
}
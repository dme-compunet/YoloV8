namespace Compunet.YoloV8.Data;

public class Segmentation : Detection, IYoloPrediction<Segmentation>
{
    public required SegmentationMask Mask { get; init; }

    static string IYoloPrediction<Segmentation>.Describe(Segmentation[] predictions) => predictions.Summary();
}
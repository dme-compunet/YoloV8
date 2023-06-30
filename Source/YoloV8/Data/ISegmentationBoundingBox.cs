namespace Compunet.YoloV8.Data;

public interface ISegmentationBoundingBox : IBoundingBox
{
    float[,] Mask { get; }
}
namespace Compunet.YoloV8.Data;

public interface ISegmentationBoundingBox : IBoundingBox
{
    IMask Mask { get; }
}
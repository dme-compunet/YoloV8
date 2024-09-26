namespace Compunet.YoloV8.Contracts.Services;

internal interface IImageAdjustmentService
{
    public ImageAdjustmentInfo Calculate(Size size);

    public Rectangle Adjust(RectangleF rectangle, ImageAdjustmentInfo adjustment);
}
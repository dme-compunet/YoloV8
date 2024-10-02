namespace Compunet.YoloV8.Contracts.Services.Plotting;

internal interface IImageContoursRecognizer
{
    public Point[][] Recognize(Image luminance);
}
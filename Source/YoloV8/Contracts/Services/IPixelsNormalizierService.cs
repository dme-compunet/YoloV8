namespace Compunet.YoloV8.Contracts.Services;

internal interface IPixelsNormalizerService
{
    public void NormalizerPixelsToTensor(Image<Rgb24> image, DenseTensor<float> tensor, Vector<int> padding);
}
namespace Compunet.YoloV8.Contracts.Services;

internal interface IPixelsNormalizerService
{
    public void NormalizerPixelsToTensor(Image<Rgb24> image, MemoryTensor<float> tensor, Vector<int> padding);
}
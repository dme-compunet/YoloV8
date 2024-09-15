namespace Compunet.YoloV8.Contracts.Services;

internal interface IPreprocessService
{
    public void ProcessImageToTensor(Image<Rgb24> image, DenseTensor<float> tensor, Vector<int> padding);
}
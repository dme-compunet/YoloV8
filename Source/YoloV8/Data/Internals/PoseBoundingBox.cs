namespace Compunet.YoloV8.Data;

internal class PoseBoundingBox(YoloV8Class name,
                               Rectangle bounds,
                               float confidence,
                               IReadOnlyList<Keypoint> keypoints) : BoundingBox(name, bounds, confidence), IPoseBoundingBox
{
    public IReadOnlyList<IKeypoint> Keypoints { get; } = keypoints;

    public IKeypoint? GetKeypoint(int index)
    {
        return Keypoints.SingleOrDefault(x => x.Index == index);
    }
}
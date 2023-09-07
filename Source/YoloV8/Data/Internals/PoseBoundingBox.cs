namespace Compunet.YoloV8.Data;

internal class PoseBoundingBox : BoundingBox, IPoseBoundingBox
{
    public IReadOnlyList<IKeypoint> Keypoints { get; }

    public PoseBoundingBox(YoloV8Class name,
                           Rectangle bounds,
                           float confidence,
                           IReadOnlyList<Keypoint> keypoints) : base(name, bounds, confidence)
    {
        Keypoints = keypoints;
    }

    public IKeypoint? GetKeypoint(int index)
    {
        return Keypoints.SingleOrDefault(x => x.Index == index);
    }
}
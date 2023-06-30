namespace Compunet.YoloV8.Data;

public interface IPoseBoundingBox : IBoundingBox
{
    IReadOnlyList<IKeypoint> Keypoints { get; }

    IKeypoint? GetKeypoint(int id);
}
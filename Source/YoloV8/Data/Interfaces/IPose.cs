namespace Compunet.YoloV8.Data;

public interface IPose
{
    Rectangle Rectangle { get; }

    float Confidence { get; }

    IReadOnlyList<IKeypoint> Keypoints { get; }
}
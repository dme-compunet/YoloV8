using Compunet.YoloV8.Parsing;

namespace YoloV8.Tests;

public class NonMaxSuppressionTests
{
    [Fact]
    public void NonMaxSuppressionBasicTest()
    {
        var nonMaxSuppression = new NonMaxSuppressionService();

        var classA = new YoloName(0, "a");
        var classB = new YoloName(1, "b");

        RawBoundingBox[] boxes =
        [
            new RawBoundingBox
            {
                Index = 0,
                Name = classA,
                Bounds = new Rectangle(0, 0, 50, 50),
                Confidence = .8f
            },
            new RawBoundingBox
            {
                Index = 1,
                Name = classA,
                Bounds = new Rectangle(0, 0, 50, 50),
                Confidence = .9f
            },
            new RawBoundingBox
            {
                Index = 2,
                Name = classB,
                Bounds = new Rectangle(0, 0, 50, 50),
                Confidence = .9f
            },
            new RawBoundingBox
            {
                Index = 3,
                Name = classA,
                Bounds = new Rectangle(50, 50, 50, 50),
                Confidence = .5f
            },
        ];

        var selected = nonMaxSuppression.Suppress(boxes.AsSpan(), .5f);

        Assert.Equal([1, 2, 3], selected.Select(x => x.Index).Order());
    }
}
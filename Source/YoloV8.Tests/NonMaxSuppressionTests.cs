namespace YoloV8.Tests;

public class NonMaxSuppressionTests
{
    [Fact]
    public void NonMaxSuppressionBasicTest()
    {
        var classA = new YoloV8Class(0, "a");
        var classB = new YoloV8Class(1, "b");

        IndexedBoundingBox[] boxes =
        [
            new IndexedBoundingBox
            {
                Index = 0,
                Class = classA,
                Bounds = new Rectangle(0, 0, 50, 50),
                Confidence = .8f
            },
            new IndexedBoundingBox
            {
                Index = 1,
                Class = classA,
                Bounds = new Rectangle(0, 0, 50, 50),
                Confidence = .9f
            },
            new IndexedBoundingBox
            {
                Index = 2,
                Class = classB,
                Bounds = new Rectangle(0, 0, 50, 50),
                Confidence = .9f
            },
            new IndexedBoundingBox
            {
                Index = 3,
                Class = classA,
                Bounds = new Rectangle(50, 50, 50, 50),
                Confidence = .5f
            },
        ];

        var selected = NonMaxSuppressionHelper.Suppress(boxes, .5f);

        Assert.Equal([1, 2, 3], selected.Select(x => x.Index).Order());
    }
}
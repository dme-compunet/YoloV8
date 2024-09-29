namespace YoloV8.Tests;

public partial class YoloPredictorTest2
{
    [Fact]
    public void PredictorTaskValidationTest()
    {
        var path = "./images/bus.jpg";

        Assert.Throws<InvalidOperationException>(() => Predictors.Pose.Detect(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Pose.DetectObb(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Pose.Segment(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Pose.Classify(path));

        Assert.Throws<InvalidOperationException>(() => Predictors.Detection.Pose(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Detection.DetectObb(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Detection.Segment(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Detection.Classify(path));

        // TODO: Add OBB

        Assert.Throws<InvalidOperationException>(() => Predictors.Segmentation.Pose(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Segmentation.Detect(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Segmentation.DetectObb(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Segmentation.Classify(path));

        Assert.Throws<InvalidOperationException>(() => Predictors.Classification.Pose(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Classification.Detect(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Classification.DetectObb(path));
        Assert.Throws<InvalidOperationException>(() => Predictors.Classification.Segment(path));
    }

    [Theory]
    [InlineData("./images/bus.jpg", "./TestAssets/detection.bus.txt")]
    [InlineData("./images/sports.jpg", "./TestAssets/detection.sports.txt")]
    public void Detection(string imagePath, string jsonPath)
    {
        var jsonData = File.ReadAllText(jsonPath);
        var expectedResult = JsonSerializer.Deserialize<Detection[]>(jsonData) ?? throw new JsonException();
        var actualResult = Predictors.Detection.Detect(imagePath).ToArray();

        Assert.Equal(expectedResult.Length, actualResult.Length);

        for (var i = 0; i < actualResult.Length; i++)
        {
            var expected = expectedResult[i];
            var actual = actualResult[i];

            Assert.Equal($"{expected.Name}", $"{actual.Name}");
            Assert.Equal(expected.Confidence, actual.Confidence);
            Assert.Equal(expected.Bounds, actual.Bounds);
        }
    }

    [JsonSerializable(typeof(Detection[]))]
    [JsonSourceGenerationOptions(WriteIndented = true)]
    partial class PredictionJsonContext : JsonSerializerContext { }
}
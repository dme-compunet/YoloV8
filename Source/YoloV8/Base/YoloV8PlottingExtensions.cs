using Path = System.IO.Path;

namespace Compunet.YoloV8;

public static class YoloV8PlottingExtensions
{
    #region TaskAndSave Sync

    public static PoseResult PoseAndSave(this YoloV8Predictor predictor, string path, string? output = null, YoloV8Configuration? configuration = null, PosePlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = predictor.Pose(image, configuration);

        using var plotted = result.PlotImage(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        plotted.Save(output);

        return result;
    }

    public static DetectionResult DetectAndSave(this YoloV8Predictor predictor, string path, string? output = null, YoloV8Configuration? configuration = null, DetectionPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = predictor.Detect(image, configuration);

        using var plotted = result.PlotImage(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        plotted.Save(output);

        return result;
    }

    public static ObbDetectionResult DetectObbAndSave(this YoloV8Predictor predictor, string path, string? output = null, YoloV8Configuration? configuration = null, DetectionPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = predictor.DetectObb(image, configuration);

        using var plotted = result.PlotImage(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        plotted.Save(output);

        return result;
    }

    public static SegmentationResult SegmentAndSave(this YoloV8Predictor predictor, string path, string? output = null, YoloV8Configuration? configuration = null, SegmentationPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = predictor.Segment(image, configuration);

        using var plotted = result.PlotImage(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        plotted.Save(output);

        return result;
    }

    public static ClassificationResult ClassifyAndSave(this YoloV8Predictor predictor, string path, string? output = null, YoloV8Configuration? configuration = null, ClassificationPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = predictor.Classify(image, configuration);

        using var plotted = result.PlotImage(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        plotted.Save(output);

        return result;
    }

    #endregion

    #region TaskAndSaveAsync

    public static async Task<PoseResult> PoseAndSaveAsync(this YoloV8Predictor predictor,
                                                          string path,
                                                          string? output = null,
                                                          YoloV8Configuration? configuration = null,
                                                          PosePlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = await predictor.PoseAsync(image, configuration);

        using var plotted = await result.PlotImageAsync(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        await plotted.SaveAsync(output);

        return result;
    }

    public static async Task<DetectionResult> DetectAndSaveAsync(this YoloV8Predictor predictor,
                                                                 string path,
                                                                 string? output = null,
                                                                 YoloV8Configuration? configuration = null,
                                                                 DetectionPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = await predictor.DetectAsync(image, configuration);

        using var plotted = await result.PlotImageAsync(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        await plotted.SaveAsync(output);

        return result;
    }


    public static async Task<ObbDetectionResult> DetectObbAndSaveAsync(this YoloV8Predictor predictor,
                                                                       string path,
                                                                       string? output = null,
                                                                       YoloV8Configuration? configuration = null,
                                                                       DetectionPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = await predictor.DetectObbAsync(image, configuration);

        using var plotted = await result.PlotImageAsync(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        await plotted.SaveAsync(output);

        return result;
    }

    public static async Task<SegmentationResult> SegmentAndSaveAsync(this YoloV8Predictor predictor,
                                                                     string path,
                                                                     string? output = null,
                                                                     YoloV8Configuration? configuration = null,
                                                                     SegmentationPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = await predictor.SegmentAsync(image, configuration);

        using var plotted = await result.PlotImageAsync(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        await plotted.SaveAsync(output);

        return result;
    }

    public static async Task<ClassificationResult> ClassifyAndSaveAsync(this YoloV8Predictor predictor,
                                                                        string path,
                                                                        string? output = null,
                                                                        YoloV8Configuration? configuration = null,
                                                                        ClassificationPlottingOptions? options = null)
    {
        using var image = Image.Load(path);

        var result = await predictor.ClassifyAsync(image, configuration);

        using var plotted = await result.PlotImageAsync(image, options);

        output ??= CreateImageOutputPath(path, predictor.Metadata.Task);

        await plotted.SaveAsync(output);

        return result;
    }

    #endregion

    #region PredictAndSave

    public static YoloV8Result PredictAndSave(this YoloV8Predictor predictor, string path, string? output = null, YoloV8Configuration? configuration = null, PlottingOptions? options = null)
    {
        return predictor.Metadata.Task switch
        {
            YoloV8Task.Pose => PoseAndSave(predictor, path, output, configuration, options as PosePlottingOptions),
            YoloV8Task.Detect => DetectAndSave(predictor, path, output, configuration, options as DetectionPlottingOptions),
            YoloV8Task.Obb => DetectObbAndSave(predictor, path, output, configuration, options as DetectionPlottingOptions),
            YoloV8Task.Segment => SegmentAndSave(predictor, path, output, configuration, options as SegmentationPlottingOptions),
            YoloV8Task.Classify => ClassifyAndSave(predictor, path, output, configuration, options as ClassificationPlottingOptions),
            _ => throw new NotSupportedException("Unsupported YOLOv8 task")
        };
    }

    public static async Task<YoloV8Result> PredictAndSaveAsync(this YoloV8Predictor predictor, string path, string? output = null, YoloV8Configuration? configuration = null, PlottingOptions? options = null)
    {
        return predictor.Metadata.Task switch
        {
            YoloV8Task.Pose => await PoseAndSaveAsync(predictor, path, output, configuration, options as PosePlottingOptions),
            YoloV8Task.Detect => await DetectAndSaveAsync(predictor, path, output, configuration, options as DetectionPlottingOptions),
            YoloV8Task.Obb => await DetectObbAndSaveAsync(predictor, path, output, configuration, options as DetectionPlottingOptions),
            YoloV8Task.Segment => await SegmentAndSaveAsync(predictor, path, output, configuration, options as SegmentationPlottingOptions),
            YoloV8Task.Classify => await ClassifyAndSaveAsync(predictor, path, output, configuration, options as ClassificationPlottingOptions),
            _ => throw new NotSupportedException("Unsupported YOLOv8 task")
        };
    }

    #endregion

    private static string CreateImageOutputPath(string path, YoloV8Task task)
    {
        var baseDirectory = Path.GetDirectoryName(path) ?? Environment.CurrentDirectory;

        var plotDirectory = Path.Combine(baseDirectory, task.ToString().ToLower());

        if (Directory.Exists(plotDirectory) == false)
        {
            Directory.CreateDirectory(plotDirectory);
        }

        var extn = Path.GetExtension(path);
        var name = Path.GetFileNameWithoutExtension(path);

        var index = 0;

        while (true)
        {
            var filename = index == 0 ? $"{name}{extn}" : $"{name}_{index}{extn}";

            var fullpath = Path.Combine(plotDirectory, filename);

            if (File.Exists(fullpath))
            {
                index++;
            }
            else
            {
                return fullpath;
            }
        }
    }
}
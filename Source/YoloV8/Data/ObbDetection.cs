﻿namespace Compunet.YoloV8.Data;

public class ObbDetection : Detection, IYoloPrediction<ObbDetection>
{
    public required float Angle { get; init; }

    static string IYoloPrediction<ObbDetection>.Describe(ObbDetection[] predictions) => predictions.Summary();
}
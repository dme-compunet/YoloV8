﻿using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Parsers;

internal readonly struct IndexedBoundingBox
{
    public int Index { get; }

    public YoloV8Class Class { get; }

    public Rectangle Bounds { get; }

    public float Confidence { get; }

    public IndexedBoundingBox(int index, YoloV8Class name, Rectangle bounds, float confidence)
    {
        Index = index;
        Class = name;
        Bounds = bounds;
        Confidence = confidence;
    }
}
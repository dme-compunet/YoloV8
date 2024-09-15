namespace Compunet.YoloV8.Memory;

internal readonly struct TensorShape(int[] shape)
{
    public int Length { get; } = GetSizeForShape(shape);

    public int[] Dimensions { get; } = shape;

    public long[] Dimensions64 { get; } = [.. shape.Select(x => (long)x)];

    private static int GetSizeForShape(ReadOnlySpan<int> shape)
    {
        var product = 1;

        for (var i = 0; i < shape.Length; i++)
        {
            var dimension = shape[i];

            if (dimension < 0)
            {
                throw new ArgumentOutOfRangeException($"Shape must not have negative elements: {dimension}");
            }

            product = checked(product * dimension);
        }

        return product;
    }
}

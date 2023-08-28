namespace Compunet.YoloV8.Extensions;

public static class ReadOnlyListExtensions
{
    public static IReadOnlyList<TResult> Select<TSource, TResult>(this IReadOnlyList<TSource> source, Func<TSource, TResult> selector)
    {
        var array = new TResult[source.Count];

        for (int i = 0; i < source.Count; i++)
        {
            var value = source[i];
            array[i] = selector(value);
        }

        return array;
    }

    public static IReadOnlyList<TResult> SelectParallely<TSource, TResult>(this IReadOnlyList<TSource> source, Func<TSource, TResult> selector)
    {
        var array = new TResult[source.Count];

        Parallel.For(0, source.Count, index =>
        {
            var value = source[index];
            array[index] = selector(value);
        });

        return array;
    }

    public static IReadOnlyList<TSource> Where<TSource>(this IReadOnlyList<TSource> source, Func<TSource, bool> predicate)
    {
        var count = default(int);

        for (int i = 0; i < source.Count; i++)
        {
            if (predicate(source[i]))
                count++;
        }

        var index = default(int);
        var result = new TSource[count];

        for (int i = 0; i < source.Count; i++)
        {
            if (predicate(source[i]))
                result[index++] = source[i];
        }

        return result;
    }
}
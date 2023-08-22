using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Compunet.YoloV8.Extensions;

public static class ReadOnlyListExtensions
{
    public static IReadOnlyList<TResult> Select<TSource, TResult>(this IReadOnlyList<TSource> values, Func<TSource, TResult> selector)
    {
        var array = new TResult[values.Count];

        for (int i = 0; i < values.Count; i++)
        {
            var value = values[i];
            array[i] = selector(value);
        }

        return array;
    }

    public static IReadOnlyList<TResult> SelectParallely<TSource, TResult>(this IReadOnlyList<TSource> values, Func<TSource, TResult> selector)
    {
        var array = new TResult[values.Count];

        Parallel.For(0, values.Count, index =>
        {
            var value = values[index];
            array[index] = selector(value);
        });

        return array;
    }
}
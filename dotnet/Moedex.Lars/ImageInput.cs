using System.Text.Json.Nodes;

namespace Moedex.Lars;

/// <summary>An immutable snapshot of one JPEG/PNG and optional context. Image answers are experimental.</summary>
public sealed class ImageInput
{
    internal byte[] Bytes { get; }
    internal JsonNode? Context { get; }

    private ImageInput(byte[] bytes, JsonNode? context)
    {
        if (bytes.Length is 0 or > 20 * 1024 * 1024)
            throw new LarsException("invalid_request", "image must contain 1 byte to 20 MiB");
        Bytes = bytes;
        Context = context?.DeepClone();
    }

    public static ImageInput FromBytes(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length is 0 or > 20 * 1024 * 1024)
            throw new LarsException("invalid_request", "image must contain 1 byte to 20 MiB");
        return new ImageInput(bytes.ToArray(), null);
    }

    public static async Task<ImageInput> FromFileAsync(string path, CancellationToken cancellationToken = default)
    {
        await using var file = File.OpenRead(path);
        if (file.Length is 0 or > 20 * 1024 * 1024)
            throw new LarsException("invalid_request", "image file must contain 1 byte to 20 MiB");
        var bytes = new byte[(int)file.Length];
        await file.ReadExactlyAsync(bytes, cancellationToken).ConfigureAwait(false);
        if (file.ReadByte() != -1) throw new LarsException("invalid_request", "image file changed while being read");
        return new ImageInput(bytes, null);
    }

    public ImageInput WithContext(JsonNode? context) => new(Bytes, context);
}

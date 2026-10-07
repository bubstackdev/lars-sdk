// P/Invoke over the C ABI (crates/lars-ffi/include/lars.h). Everything crosses as UTF-8 JSON; strings
// the library returns are freed with lars_string_free. Async calls complete a TaskCompletionSource
// from the library's worker thread, through one static callback and a GCHandle.

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;

namespace Moedex.Lars;

internal sealed class EngineHandle : SafeHandle
{
    public EngineHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    /// <summary>Closing the last handle on a pack frees its model before returning.</summary>
    protected override bool ReleaseHandle()
    {
        Native.lars_close(handle);
        return true;
    }
}

internal static unsafe partial class Native
{
    private const string Library = "lars";

    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
    }

    /// <summary>LARS_NATIVE names the library outright; otherwise the NuGet layout and normal probing apply.</summary>
    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? path)
    {
        if (name != Library) return IntPtr.Zero;
        var explicitPath = Environment.GetEnvironmentVariable("LARS_NATIVE");
        if (!string.IsNullOrEmpty(explicitPath)) return NativeLibrary.Load(explicitPath);
        return NativeLibrary.TryLoad(name, assembly, path, out var loaded) ? loaded : IntPtr.Zero;
    }

    [LibraryImport(Library)]
    internal static partial IntPtr lars_version();

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr lars_open(string pack, string? optionsJson, out IntPtr errorJson);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr lars_evaluate(EngineHandle engine, string requestJson, out IntPtr errorJson);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial ulong lars_evaluate_async(EngineHandle engine, string requestJson,
        delegate* unmanaged[Cdecl]<IntPtr, ulong, IntPtr, IntPtr, void> callback, IntPtr userData);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial ulong lars_calibrate_async(EngineHandle engine, string name, string examplesJsonl,
        double maxError, delegate* unmanaged[Cdecl]<IntPtr, ulong, IntPtr, IntPtr, void> callback, IntPtr userData);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int lars_set_calibrator(EngineHandle engine, string name, string calibratorJson, out IntPtr errorJson);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr lars_pull(string pack, out IntPtr errorJson);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int lars_set_calibrators(EngineHandle engine, string calibratorsJson, out IntPtr errorJson);

    [LibraryImport(Library)]
    internal static partial IntPtr lars_calibrators(EngineHandle engine);

    [LibraryImport(Library)]
    internal static partial IntPtr lars_status(EngineHandle engine);

    [LibraryImport(Library)]
    [return: MarshalAs(UnmanagedType.U1)]
    internal static partial bool lars_cancel(EngineHandle engine, ulong requestId);

    [LibraryImport(Library)]
    internal static partial void lars_string_free(IntPtr text);

    [LibraryImport(Library)]
    internal static partial void lars_close(IntPtr engine);

    /// <summary>A string the library returned, freed after reading.</summary>
    internal static string? Take(IntPtr text)
    {
        if (text == IntPtr.Zero) return null;
        try { return Marshal.PtrToStringUTF8(text); }
        finally { lars_string_free(text); }
    }

    internal static LarsException Error(string? json)
    {
        if (json is not null)
        {
            try
            {
                var node = JsonNode.Parse(json);
                var type = node?["error_type"]?.GetValue<string>();
                if (type is not null) return new LarsException(type, node?["message"]?.GetValue<string>() ?? json);
            }
            catch (System.Text.Json.JsonException) { }
        }
        return new LarsException("internal_error", json ?? "unknown error");
    }

    /// <summary>The text of a call that returns NULL and sets an error on failure.</summary>
    internal static string Check(IntPtr result, IntPtr errorJson)
    {
        var error = Take(errorJson);
        return Take(result) ?? throw Error(error);
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void OnReply(IntPtr userData, ulong requestId, IntPtr resultJson, IntPtr errorJson)
    {
        var handle = GCHandle.FromIntPtr(userData);
        var completion = (TaskCompletionSource<string>)handle.Target!;
        handle.Free();
        // The strings are only valid during this call: read them now.
        var result = resultJson == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(resultJson);
        var error = errorJson == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(errorJson);
        if (result is not null) completion.TrySetResult(result);
        else completion.TrySetException(Error(error));
    }

    /// <summary>The callback every async call is submitted with.</summary>
    internal static IntPtr ReplyCallback()
    {
        delegate* unmanaged[Cdecl]<IntPtr, ulong, IntPtr, IntPtr, void> callback = &OnReply;
        return (IntPtr)callback;
    }

    internal static ulong EvaluateAsync(EngineHandle engine, string request, IntPtr callback, IntPtr user) =>
        lars_evaluate_async(engine, request, (delegate* unmanaged[Cdecl]<IntPtr, ulong, IntPtr, IntPtr, void>)callback, user);

    internal static ulong CalibrateAsync(EngineHandle engine, string name, string examples, double maxError, IntPtr callback, IntPtr user) =>
        lars_calibrate_async(engine, name, examples, maxError, (delegate* unmanaged[Cdecl]<IntPtr, ulong, IntPtr, IntPtr, void>)callback, user);
}

internal static class Calls
{
    /// <summary>Submit an async call and await its one reply. Cancelling the token cancels the request if it has not started.</summary>
    internal static async Task<string> Submit(EngineHandle engine,
        Func<IntPtr, IntPtr, ulong> submit, CancellationToken cancellationToken)
    {
        // Continuations must not run on the library's worker thread, which serves every caller.
        var completion = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        var handle = GCHandle.Alloc(completion);
        var id = submit(Native.ReplyCallback(), GCHandle.ToIntPtr(handle));
        await using var registration = id == 0 || !cancellationToken.CanBeCanceled
            ? default
            : cancellationToken.Register(() =>
            {
                try { Native.lars_cancel(engine, id); }
                catch (ObjectDisposedException) { } // closed meanwhile: closing cancels it anyway
            });
        try
        {
            return await completion.Task.ConfigureAwait(false);
        }
        catch (LarsException e) when (e.Type == "cancelled" && cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }
    }

}

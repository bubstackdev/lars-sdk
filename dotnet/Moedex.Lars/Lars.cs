// Moedex.Lars: typed decisions as functions in your code.
//
//     using var lars = await Lars.OpenAsync("lars-2b");
//     var isPissed = lars.Noul("isPissed", "The customer is angry");
//     if (isPissed.Check(message)) Escalate(message);            // blocking: scripts, console apps
//     if (await isPissed.AskAsync(message)) Escalate(message);   // ASP.NET and other async code
//
// The model runs in this process. Calls queue on its worker thread; async calls complete from it
// without blocking a thread-pool thread while they wait.

using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Moedex.Lars;

/// <summary>An error from the engine. <see cref="Type"/> is invalid_request, pack_error, cancelled, api_error or internal_error.</summary>
public sealed class LarsException(string type, string message) : Exception(message)
{
    public string Type { get; } = type;
}

/// <summary>Gives a choice option (an enum member) the description the model reads, and optionally its key.</summary>
[AttributeUsage(AttributeTargets.Field)]
public sealed class OptionAttribute(string description) : Attribute
{
    public string Description { get; } = description;

    /// <summary>The key sent to the model; default the member name in snake_case.</summary>
    public string? Key { get; init; }
}

public sealed record LarsOptions
{
    /// <summary>A pack directory, or <c>name[@version]</c> from the pack cache (downloaded on first use).</summary>
    public required string Pack { get; init; }
    /// <summary>Layers on the GPU (Metal): -1 for all (the default), 0 for CPU only.</summary>
    public int? GpuLayers { get; init; }
    /// <summary>CPU threads; 0 for the number of cores.</summary>
    public int? Threads { get; init; }
    public bool? BatchSuffixes { get; init; }
    public bool? ReusePrefix { get; init; }
    public int? MaxRows { get; init; }
    public int? MaxInputTokens { get; init; }
    /// <summary>Free the model after this many idle seconds; it reloads on the next call.</summary>
    public double? IdleUnloadSeconds { get; init; }
    /// <summary>Check every pack file's sha256 on open, not only its size.</summary>
    public bool? Verify { get; init; }

    internal string ToJson()
    {
        var o = new JsonObject();
        void Put(string key, JsonNode? value) { if (value is not null) o[key] = value; }
        Put("gpu_layers", GpuLayers); Put("threads", Threads); Put("batch_suffixes", BatchSuffixes);
        Put("reuse_prefix", ReusePrefix); Put("max_rows", MaxRows); Put("max_input_tokens", MaxInputTokens);
        Put("idle_unload_s", IdleUnloadSeconds); Put("verify", Verify);
        return o.ToJsonString();
    }
}

/// <summary>An acceptance threshold certified on labeled examples: the Wilson 95% bound on the error rate. A null
/// <see cref="Threshold"/> means the labels could not certify <see cref="MaxError"/> (<see cref="Reason"/> says why); then
/// no answer is accepted until a calibration that certifies replaces it.</summary>
public sealed record Certified(double MaxError, double? Threshold, int? Accepted, int? Errors, double? Coverage,
                               double? ErrorUpper95, string? Reason);

/// <summary>A calibration's outcome. <see cref="Kept"/> is what the predicate now uses: "pack" or "fitted" (a refit on
/// these labels, when significantly better on held-out labels by 5-fold cross-validated log loss); "recal" (that,
/// rescaled with a bias per option); or "head" (plus a readout of the model's hidden state, tried from 50 labels of
/// one question). Each step is taken only when significantly better than the last. <see cref="Accuracy"/> is
/// cross-validated for "recal" and "head", else in-sample.</summary>
public sealed record CalibrationResult(int Examples, string Kept, double Accuracy, Certified? Certified);

/// <summary>What every decision carries.</summary>
public abstract record Decision
{
    /// <summary>The top probability: how likely the answer is right, once calibrated. Thresholds compare to it.</summary>
    public double Confidence { get; init; }
    /// <summary>False when this predicate has a certificate and the confidence is below its threshold, or certification was
    /// asked for and failed (no threshold): then nothing is accepted.</summary>
    public bool Accepted { get; init; } = true;
    public bool? Abstain { get; init; }
    public IReadOnlyList<(string Span, double Effect)>? Evidence { get; init; }
}

public sealed record NoulDecision : Decision
{
    public bool Value { get; init; }
    /// <summary>P(yes).</summary>
    public double P { get; init; }

    /// <summary>True when the answer is yes and accepted, so <c>if (await p.AskAsync(x))</c> reads as it should.</summary>
    public static implicit operator bool(NoulDecision d) => d.Value && d.Accepted;
}

public sealed record ChoiceDecision<T> : Decision where T : struct, Enum
{
    public T Value { get; init; }
    public IReadOnlyDictionary<T, double> Probabilities { get; init; } = new Dictionary<T, double>();
    public double? OrderSensitivity { get; init; }

    public static implicit operator T(ChoiceDecision<T> d) => d.Value;
}

public sealed record ScoreDecision : Decision
{
    /// <summary>The expected level, sum of i * P(level i); may fall between levels.</summary>
    public double Value { get; init; }
    /// <summary>The most probable level.</summary>
    public int Level { get; init; }
    public IReadOnlyList<double> Probabilities { get; init; } = [];

    public static implicit operator double(ScoreDecision d) => d.Value;
}

public sealed record MultiDecision<T> where T : struct, Enum
{
    public IReadOnlyList<T> Value { get; init; } = [];
    public IReadOnlyDictionary<T, double> Probabilities { get; init; } = new Dictionary<T, double>();
}

/// <summary>Options shared by every predicate.</summary>
public record PredicateOptions<TInput>
{
    /// <summary>What the model reads for an input: default the string itself, or the object as JSON. Pick the fields that
    /// matter, so whole records (and the personal data in them) stay out of prompts.</summary>
    public Func<TInput, JsonNode?>? State { get; init; }
    public double? AbstainMargin { get; init; }
    public bool Explain { get; init; }
    /// <summary>Names the version of what <see cref="State"/> reads, e.g. "tickets-v2". Saved calibrations record it, and
    /// loading one for a predicate whose version differs is refused: change the fields a predicate reads, bump the
    /// version, and calibrations fitted on the old fields can't be used by mistake. The selector itself can't be saved,
    /// so without a version a change to it goes unnoticed.</summary>
    public string? StateVersion { get; init; }
}

public sealed record NoulOptions<TInput> : PredicateOptions<TInput>
{
    /// <summary>P(yes) at or above which the answer is yes; default 0.5.</summary>
    public double Threshold { get; init; } = 0.5;
}

public sealed record ChoiceOptions<TInput, T> : PredicateOptions<TInput> where T : struct, Enum
{
    /// <summary>Returned by <c>Pick</c> when the answer is not accepted under a certified threshold.</summary>
    public T? Fallback { get; init; }
    public int Permutations { get; init; }
}

/// <summary>Something <c>Lars.AskAsync</c> can put to a request.</summary>
public abstract class Predicate<TInput>
{
    private protected Predicate(Lars lars, string name, JsonObject question, PredicateOptions<TInput> options)
    {
        Lars = lars;
        Name = name;
        Question = question;
        Options = options;
    }

    internal Lars Lars { get; }
    public string Name { get; }
    /// <summary>The System One question this predicate asks.</summary>
    public JsonObject Question { get; }
    internal PredicateOptions<TInput> Options { get; }

    internal JsonNode? StateOf(TInput input) => Options.State is { } state ? state(input) : Lars.DefaultState(input);

    internal virtual JsonObject Extensions()
    {
        var ext = new JsonObject();
        if (Options.AbstainMargin is { } margin) ext["abstain_margin"] = margin;
        if (Options.Explain) ext["explain"] = true;
        if (Lars.IsCalibrated(Name)) ext["calibrators"] = new JsonObject { [Name] = Name };
        return ext;
    }

    internal async Task<JsonNode> AnswerAsync(TInput input, CancellationToken cancellationToken)
    {
        var request = new JsonObject
        {
            ["state"] = StateOf(input)?.DeepClone(),
            ["questions"] = new JsonObject { [Name] = Question.DeepClone() },
            ["lars"] = Extensions(),
        };
        var response = await Lars.EvaluateAsync(request, cancellationToken).ConfigureAwait(false);
        return response["answers"]![Name]!;
    }

    internal JsonNode Answer(TInput input)
    {
        var request = new JsonObject
        {
            ["state"] = StateOf(input)?.DeepClone(),
            ["questions"] = new JsonObject { [Name] = Question.DeepClone() },
            ["lars"] = Extensions(),
        };
        return Lars.Evaluate(request)["answers"]![Name]!;
    }

    /// <summary>Fit this predicate's own calibrator; labels are rendered as the eval format expects.</summary>
    private protected async Task<CalibrationResult> CalibrateCore<TLabel>(IEnumerable<(TInput Input, TLabel Label)> examples,
        Func<TLabel, string> label, double? maxError, CancellationToken cancellationToken)
    {
        var jsonl = new StringBuilder();
        foreach (var (input, l) in examples)
        {
            var row = new JsonObject { ["state"] = StateOf(input)?.DeepClone(), ["question"] = Question.DeepClone(), ["label"] = label(l) };
            jsonl.Append(row.ToJsonString()).Append('\n');
        }
        return await Lars.CalibrateAsync(Name, jsonl.ToString(), maxError, cancellationToken).ConfigureAwait(false);
    }

    private protected static IReadOnlyList<(string, double)>? EvidenceOf(JsonNode answer) =>
        answer["evidence"]?.AsArray().Select(e => (e!["span"]!.GetValue<string>(), e["effect"]!.GetValue<double>())).ToList();

    internal abstract object DecideFrom(JsonNode answer);
}

/// <summary>A yes/no predicate.</summary>
public sealed class Noul<TInput> : Predicate<TInput>
{
    internal Noul(Lars lars, string name, JsonObject question, NoulOptions<TInput> options) : base(lars, name, question, options) =>
        Threshold = options.Threshold;

    private double Threshold { get; }

    /// <summary>Blocks until answered: true when the claim holds and the answer is accepted.</summary>
    public bool Check(TInput input) => Decide(Answer(input));

    public async Task<NoulDecision> AskAsync(TInput input, CancellationToken cancellationToken = default) =>
        Decide(await AnswerAsync(input, cancellationToken).ConfigureAwait(false));

    public Task<CalibrationResult> CalibrateAsync(IEnumerable<(TInput Input, bool Label)> examples, double? maxError = null,
        CancellationToken cancellationToken = default) =>
        CalibrateCore(examples, l => l ? "yes" : "no", maxError, cancellationToken);

    /// <summary>Use the predicate wherever a <c>Func&lt;TInput, bool&gt;</c> goes (a blocking call).</summary>
    public static implicit operator Func<TInput, bool>(Noul<TInput> noul) => noul.Check;

    internal NoulDecision Decide(JsonNode answer)
    {
        var p = answer["noul"]!.GetValue<double>();
        var confidence = Math.Max(p, 1 - p);
        return new NoulDecision
        {
            Value = p >= Threshold, P = p, Confidence = confidence, Accepted = Lars.Accepts(Name, confidence),
            Abstain = answer["abstain"]?.GetValue<bool>(), Evidence = EvidenceOf(answer),
        };
    }

    internal override object DecideFrom(JsonNode answer) => Decide(answer);
}

/// <summary>Pick one option of an enum.</summary>
public sealed class Choice<TInput, T> : Predicate<TInput> where T : struct, Enum
{
    private readonly Dictionary<string, T> _byKey;
    private readonly T? _fallback;
    private readonly int _permutations;

    internal Choice(Lars lars, string name, JsonObject question, Dictionary<string, T> byKey, ChoiceOptions<TInput, T> options)
        : base(lars, name, question, options)
    {
        _byKey = byKey;
        _fallback = options.Fallback;
        _permutations = options.Permutations;
    }

    /// <summary>Blocks until answered; returns the fallback, if one is set, when the answer is not accepted.</summary>
    public T Pick(TInput input)
    {
        var d = Decide(Answer(input));
        return !d.Accepted && _fallback is { } f ? f : d.Value;
    }

    public async Task<ChoiceDecision<T>> AskAsync(TInput input, CancellationToken cancellationToken = default) =>
        Decide(await AnswerAsync(input, cancellationToken).ConfigureAwait(false));

    public Task<CalibrationResult> CalibrateAsync(IEnumerable<(TInput Input, T Label)> examples, double? maxError = null,
        CancellationToken cancellationToken = default) =>
        CalibrateCore(examples, l => _byKey.First(kv => kv.Value.Equals(l)).Key, maxError, cancellationToken);

    internal override JsonObject Extensions()
    {
        var ext = base.Extensions();
        if (_permutations > 0) ext["permutations"] = _permutations;
        return ext;
    }

    internal ChoiceDecision<T> Decide(JsonNode answer)
    {
        var probabilities = answer["probabilities"]!.AsObject().ToDictionary(kv => _byKey[kv.Key], kv => kv.Value!.GetValue<double>());
        var confidence = probabilities.Values.Max();
        return new ChoiceDecision<T>
        {
            Value = _byKey[answer["choice"]!.GetValue<string>()], Probabilities = probabilities, Confidence = confidence,
            Accepted = Lars.Accepts(Name, confidence), OrderSensitivity = answer["order_sensitivity"]?.GetValue<double>(),
            Abstain = answer["abstain"]?.GetValue<bool>(), Evidence = EvidenceOf(answer),
        };
    }

    internal override object DecideFrom(JsonNode answer) => Decide(answer);
}

/// <summary>Place the input on ordered levels, lowest first.</summary>
public sealed class Score<TInput> : Predicate<TInput>
{
    private readonly int _levels;

    internal Score(Lars lars, string name, JsonObject question, int levels, PredicateOptions<TInput> options)
        : base(lars, name, question, options) => _levels = levels;

    /// <summary>Blocks until answered: the expected level.</summary>
    public double Rate(TInput input) => Decide(Answer(input)).Value;

    public async Task<ScoreDecision> AskAsync(TInput input, CancellationToken cancellationToken = default) =>
        Decide(await AnswerAsync(input, cancellationToken).ConfigureAwait(false));

    /// <summary>Labels are level indices, 0 for the first level.</summary>
    public Task<CalibrationResult> CalibrateAsync(IEnumerable<(TInput Input, int Label)> examples, double? maxError = null,
        CancellationToken cancellationToken = default) =>
        CalibrateCore(examples, l => l.ToString(CultureInfo.InvariantCulture), maxError, cancellationToken);

    internal ScoreDecision Decide(JsonNode answer)
    {
        var probabilities = Enumerable.Range(0, _levels)
            .Select(i => answer["probabilities"]![i.ToString(CultureInfo.InvariantCulture)]?.GetValue<double>() ?? 0).ToList();
        var top = probabilities.Max();
        return new ScoreDecision
        {
            Value = answer["score"]!.GetValue<double>(), Level = probabilities.IndexOf(top), Probabilities = probabilities,
            Confidence = top, Accepted = Lars.Accepts(Name, top), Abstain = answer["abstain"]?.GetValue<bool>(),
            Evidence = EvidenceOf(answer),
        };
    }

    internal override object DecideFrom(JsonNode answer) => Decide(answer);
}

/// <summary>Every option of an enum that applies.</summary>
public sealed class Multi<TInput, T> : Predicate<TInput> where T : struct, Enum
{
    private readonly Dictionary<string, T> _byKey;

    internal Multi(Lars lars, string name, JsonObject question, Dictionary<string, T> byKey, PredicateOptions<TInput> options)
        : base(lars, name, question, options) => _byKey = byKey;

    /// <summary>Blocks until answered: the options whose probability is at least 0.5.</summary>
    public IReadOnlyList<T> Select(TInput input) => Decide(Answer(input)).Value;

    public async Task<MultiDecision<T>> AskAsync(TInput input, CancellationToken cancellationToken = default) =>
        Decide(await AnswerAsync(input, cancellationToken).ConfigureAwait(false));

    internal MultiDecision<T> Decide(JsonNode answer) => new()
    {
        Value = answer["selected"]!.AsArray().Select(k => _byKey[k!.GetValue<string>()]).ToList(),
        Probabilities = answer["probabilities"]!.AsObject().ToDictionary(kv => _byKey[kv.Key], kv => kv.Value!.GetValue<double>()),
    };

    internal override object DecideFrom(JsonNode answer) => Decide(answer);
}

/// <summary>The answers of <c>Lars.AskAsync</c>, one per predicate.</summary>
public sealed class Answers
{
    private readonly Dictionary<string, object> _decisions;

    internal Answers(Dictionary<string, object> decisions) => _decisions = decisions;

    public NoulDecision Of<TInput>(Noul<TInput> p) => (NoulDecision)_decisions[p.Name];
    public ChoiceDecision<T> Of<TInput, T>(Choice<TInput, T> p) where T : struct, Enum => (ChoiceDecision<T>)_decisions[p.Name];
    public ScoreDecision Of<TInput>(Score<TInput> p) => (ScoreDecision)_decisions[p.Name];
    public MultiDecision<T> Of<TInput, T>(Multi<TInput, T> p) where T : struct, Enum => (MultiDecision<T>)_decisions[p.Name];
}

/// <summary>An open pack. Thread-safe; dispose it to release the model.</summary>
public sealed class Lars : IDisposable, IAsyncDisposable
{
    private readonly EngineHandle _handle;
    private readonly Dictionary<string, string> _questions = [];
    private readonly HashSet<string> _calibrated = [];
    private readonly Dictionary<string, Certified> _certified = [];
    private readonly Lock _lock = new();

    private Lars(EngineHandle handle)
    {
        _handle = handle;
        Pack = JsonNode.Parse(Native.Take(Native.lars_status(handle))!)!["pack"]!.AsObject();
    }

    /// <summary>The pack: name, version, directory, layout, llama.cpp build.</summary>
    public JsonObject Pack { get; }

    public static string Version => System.Runtime.InteropServices.Marshal.PtrToStringUTF8(Native.lars_version())!;

    /// <summary>Open a pack and load its model; blocks while it loads.</summary>
    public static Lars Open(LarsOptions options)
    {
        var engine = Native.lars_open(options.Pack, options.ToJson(), out var error);
        var message = Native.Take(error);
        if (engine == IntPtr.Zero) throw Native.Error(message);
        var handle = new EngineHandle();
        System.Runtime.InteropServices.Marshal.InitHandle(handle, engine);
        return new Lars(handle);
    }

    public static Lars Open(string pack) => Open(new LarsOptions { Pack = pack });

    /// <summary>Open a pack, loading the model on a thread-pool thread.</summary>
    public static Task<Lars> OpenAsync(LarsOptions options) => Task.Run(() => Open(options));

    public static Task<Lars> OpenAsync(string pack) => OpenAsync(new LarsOptions { Pack = pack });

    // ------------------------------------------------------------------ predicates

    public Noul<string> Noul(string name, string instructions, double threshold = 0.5) =>
        Noul(name, instructions, new NoulOptions<string> { Threshold = threshold });

    public Noul<TInput> Noul<TInput>(string name, JsonNode? instructions, NoulOptions<TInput> options) =>
        new(this, name, Register(name, new JsonObject { ["type"] = "noul", ["instructions"] = instructions?.DeepClone() }, options.StateVersion), options);

    public Choice<string, T> Choice<T>(string name, string instructions) where T : struct, Enum =>
        Choice(name, instructions, new ChoiceOptions<string, T>());

    public Choice<TInput, T> Choice<TInput, T>(string name, JsonNode? instructions, ChoiceOptions<TInput, T> options) where T : struct, Enum
    {
        var (criteria, byKey) = Options<T>();
        var question = new JsonObject { ["type"] = "choice", ["instructions"] = instructions?.DeepClone(), ["criteria"] = criteria };
        return new Choice<TInput, T>(this, name, Register(name, question, options.StateVersion), byKey, options);
    }

    public Score<string> Score(string name, string instructions, params string[] levels) =>
        Score(name, instructions, levels, new PredicateOptions<string>());

    public Score<TInput> Score<TInput>(string name, JsonNode? instructions, IReadOnlyList<string> levels, PredicateOptions<TInput> options)
    {
        var question = new JsonObject
        {
            ["type"] = "score", ["instructions"] = instructions?.DeepClone(),
            ["criteria"] = new JsonArray(levels.Select(l => (JsonNode)l).ToArray()),
        };
        return new Score<TInput>(this, name, Register(name, question, options.StateVersion), levels.Count, options);
    }

    public Multi<string, T> Multi<T>(string name, string instructions) where T : struct, Enum =>
        Multi(name, instructions, new PredicateOptions<string>(), default(T));

    public Multi<TInput, T> Multi<TInput, T>(string name, JsonNode? instructions, PredicateOptions<TInput> options, T _ = default)
        where T : struct, Enum
    {
        var (criteria, byKey) = Options<T>();
        var question = new JsonObject { ["type"] = "multi", ["instructions"] = instructions?.DeepClone(), ["criteria"] = criteria };
        return new Multi<TInput, T>(this, name, Register(name, question, options.StateVersion), byKey, options);
    }

    /// <summary>Ask several predicates about one input in one request; the model reads the input once.</summary>
    public async Task<Answers> AskAsync<TInput>(TInput input, IReadOnlyList<Predicate<TInput>> predicates,
        IReadOnlyList<(string Kind, string[] Questions)>? constraints = null, CancellationToken cancellationToken = default)
    {
        if (predicates.Count == 0) throw new LarsException("invalid_request", "AskAsync needs at least one predicate");
        var questions = new JsonObject();
        var calibrators = new JsonObject();
        foreach (var p in predicates)
        {
            questions[p.Name] = p.Question.DeepClone();
            if (IsCalibrated(p.Name)) calibrators[p.Name] = p.Name;
        }
        var ext = new JsonObject { ["calibrators"] = calibrators };
        if (constraints is not null)
        {
            ext["constraints"] = new JsonArray(constraints.Select(c => (JsonNode)new JsonObject
            {
                ["kind"] = c.Kind, ["questions"] = new JsonArray(c.Questions.Select(q => (JsonNode)q).ToArray()),
            }).ToArray());
        }
        var request = new JsonObject { ["state"] = predicates[0].StateOf(input)?.DeepClone(), ["questions"] = questions, ["lars"] = ext };
        var answers = (await EvaluateAsync(request, cancellationToken).ConfigureAwait(false))["answers"]!;
        return new Answers(predicates.ToDictionary(p => p.Name, p => p.DecideFrom(answers[p.Name]!)));
    }

    public Task<Answers> AskAsync<TInput>(TInput input, params Predicate<TInput>[] predicates) => AskAsync(input, (IReadOnlyList<Predicate<TInput>>)predicates);

    // ------------------------------------------------------------------ raw calls

    /// <summary>Send a raw System One request (blocking).</summary>
    public JsonNode Evaluate(JsonObject request) =>
        JsonNode.Parse(Native.Check(Native.lars_evaluate(_handle, request.ToJsonString(), out var error), error))!;

    /// <summary>Send a raw System One request.</summary>
    public async Task<JsonNode> EvaluateAsync(JsonObject request, CancellationToken cancellationToken = default)
    {
        var json = request.ToJsonString();
        var text = await Calls.Submit(_handle, (callback, user) => Native.EvaluateAsync(_handle, json, callback, user), cancellationToken)
            .ConfigureAwait(false);
        return JsonNode.Parse(text)!;
    }

    internal async Task<CalibrationResult> CalibrateAsync(string name, string jsonl, double? maxError, CancellationToken cancellationToken)
    {
        var text = await Calls.Submit(_handle, (callback, user) => Native.CalibrateAsync(_handle, name, jsonl, maxError ?? 0, callback, user),
            cancellationToken).ConfigureAwait(false);
        var raw = JsonNode.Parse(text)!;
        Certified? certified = null;
        if (raw["certified"] is JsonObject c)
        {
            certified = new Certified(c["max_error"]!.GetValue<double>(), c["threshold"]?.GetValue<double>(), c["accepted"]?.GetValue<int>(),
                c["errors"]?.GetValue<int>(), c["coverage"]?.GetValue<double>(), c["error_upper_95"]?.GetValue<double>(),
                c["reason"]?.GetValue<string>());
        }
        lock (_lock)
        {
            _calibrated.Add(name);
            if (certified is null) _certified.Remove(name);
            else _certified[name] = certified;
        }
        return new CalibrationResult(raw["examples"]!.GetValue<int>(), raw["kept"]!.GetValue<string>(), raw["accuracy"]!.GetValue<double>(),
            certified);
    }

    // ------------------------------------------------------------------ calibrations

    /// <summary>Every calibrator fitted or loaded on this handle, with the question each was fitted for and its certified
    /// threshold, as JSON: the file format all three SDKs save and load.</summary>
    public JsonObject Calibrations()
    {
        var certified = new JsonObject();
        var questions = new JsonObject();
        var stateVersions = new JsonObject();
        var calibrators = JsonNode.Parse(Native.Take(Native.lars_calibrators(_handle))!)!.AsObject();
        lock (_lock)
        {
            foreach (var (name, _) in calibrators)
            {
                if (!_questions.TryGetValue(name, out var key)) continue;
                var bound = JsonNode.Parse(key)!;
                questions[name] = bound["question"]!.DeepClone();
                if (bound["state_version"] is JsonValue version) stateVersions[name] = version.DeepClone();
            }
            foreach (var (name, c) in _certified)
            {
                certified[name] = new JsonObject
                {
                    ["maxError"] = c.MaxError, ["threshold"] = c.Threshold, ["accepted"] = c.Accepted, ["errors"] = c.Errors,
                    ["coverage"] = c.Coverage, ["errorUpper95"] = c.ErrorUpper95, ["reason"] = c.Reason,
                };
            }
        }
        return new JsonObject
        {
            ["format"] = 2,
            ["pack"] = new JsonObject { ["name"] = Pack["name"]!.DeepClone(), ["version"] = Pack["version"]!.DeepClone() },
            ["calibrators"] = calibrators,
            ["questions"] = questions,
            ["stateVersions"] = stateVersions,
            ["certified"] = certified,
        };
    }

    public void SaveCalibrations(string path) =>
        File.WriteAllText(path, Calibrations().ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");

    /// <summary>Load calibrations saved from this pack (the JSON, or a file path), by any of the SDKs. Each replaces the
    /// predicate's calibrator and its certificate (clearing it if the file has none); other predicates keep theirs. A
    /// calibration is bound to the question it was fitted for: a predicate of that name asking anything else is refused,
    /// here or when defined later. The whole file is checked first, so one that is refused changes nothing.</summary>
    public void LoadCalibrations(JsonObject saved)
    {
        static LarsException Invalid(string message) => new("invalid_request", message);
        if (saved["format"] is not JsonValue format || !format.TryGetValue<int>(out var f) || f != 2)
            throw Invalid("these calibrations are not format 2; calibrate again with this SDK");
        var (packName, packVersion) = (saved["pack"]?["name"]?.ToString(), saved["pack"]?["version"]?.ToString());
        if (packName != Pack["name"]?.ToString() || packVersion != Pack["version"]?.ToString())
            throw Invalid($"these calibrations are for pack {packName}@{packVersion}, not {Pack["name"]}@{Pack["version"]}");
        if (saved["calibrators"] is not JsonObject calibrators) throw Invalid("no calibrators to load");
        var incoming = new List<(string Name, string Question, Certified? Certified)>();
        lock (_lock)
        {
            foreach (var (name, _) in calibrators)
            {
                if (saved["questions"]?[name] is not JsonObject question)
                    throw Invalid($"the saved calibration for \"{name}\" records no question");
                string? version = saved["stateVersions"]?[name] switch
                {
                    null => null,
                    JsonValue v when v.TryGetValue<string>(out var text) => text,
                    _ => throw Invalid($"the state version for \"{name}\" is not a string"),
                };
                var key = Binding(question, version);
                if (_questions.TryGetValue(name, out var existing) && existing != key)
                    throw Invalid($"the saved calibration for \"{name}\" was fitted for a different question or state version than the predicate of that name here");
                incoming.Add((name, key, SavedCertified(name, saved["certified"]?[name])));
            }
        }
        if (Native.lars_set_calibrators(_handle, calibrators.ToJsonString(), out var error) != 0) throw Native.Error(Native.Take(error));
        lock (_lock)
        {
            foreach (var (name, question, certified) in incoming)
            {
                _questions[name] = question;
                _calibrated.Add(name);
                if (certified is null) _certified.Remove(name);
                else _certified[name] = certified;
            }
        }
    }

    /// <summary>A certificate as the calibrations file records it, checked field by field.</summary>
    private static Certified? SavedCertified(string name, JsonNode? c)
    {
        if (c is null) return null;
        LarsException Bad(string field) => new("invalid_request", $"the certificate for \"{name}\" has a bad {field}");
        if (c is not JsonObject o) throw Bad("entry");
        double? Number(string field)
        {
            var v = o[field];
            if (v is null) return null;
            if (v is JsonValue value && value.TryGetValue<double>(out var d) && double.IsFinite(d)) return d;
            throw Bad(field);
        }
        int? Count(string field) => Number(field) is { } d ? (d == Math.Floor(d) && d >= 0 ? (int)d : throw Bad(field)) : null;
        string? reason = o["reason"] switch
        {
            null => null,
            JsonValue v when v.TryGetValue<string>(out var text) => text,
            _ => throw Bad("reason"),
        };
        return new Certified(Number("maxError") ?? throw Bad("maxError"), Number("threshold"), Count("accepted"), Count("errors"),
            Number("coverage"), Number("errorUpper95"), reason);
    }

    public void LoadCalibrations(string path) => LoadCalibrations(JsonNode.Parse(File.ReadAllText(path))!.AsObject());

    /// <summary>Pack, model, loaded or not, queue depth, calibrators, versions.</summary>
    public JsonObject Status() => JsonNode.Parse(Native.Take(Native.lars_status(_handle))!)!.AsObject();

    public void Dispose() => _handle.Dispose();

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    // ------------------------------------------------------------------ internals

    internal bool IsCalibrated(string name)
    {
        lock (_lock) return _calibrated.Contains(name);
    }

    internal bool Accepts(string name, double confidence)
    {
        // No certificate: everything is accepted. A failed one (no threshold): nothing is.
        lock (_lock) return !_certified.TryGetValue(name, out var c) || (c.Threshold is { } t && confidence >= t);
    }

    internal static JsonNode? DefaultState<TInput>(TInput input) => input switch
    {
        null => null,
        string s => JsonValue.Create(s),
        JsonNode node => node,
        _ => throw new LarsException("invalid_request",
            $"give a State selector to turn a {typeof(TInput).Name} into text or JSON (PredicateOptions.State)"),
    };

    private JsonObject Register(string name, JsonObject question, string? stateVersion)
    {
        var key = Binding(question, stateVersion);
        lock (_lock)
        {
            if (_questions.TryGetValue(name, out var existing) && existing != key)
                throw new LarsException("invalid_request",
                    $"a predicate named \"{name}\" already asks a different question or reads a different state version");
            _questions[name] = key;
        }
        return question;
    }

    /// <summary>What a predicate's name is bound to: its question and the version of its state selector.</summary>
    private static string Binding(JsonObject question, string? stateVersion) =>
        new JsonObject { ["question"] = question.DeepClone(), ["state_version"] = stateVersion }.ToJsonString();

    /// <summary>An enum's members as choice criteria: snake_case keys (or [Option(Key=...)]) and their descriptions.</summary>
    private static (JsonObject Criteria, Dictionary<string, T> ByKey) Options<T>() where T : struct, Enum
    {
        var criteria = new JsonObject();
        var byKey = new Dictionary<string, T>();
        foreach (var field in typeof(T).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            var option = field.GetCustomAttribute<OptionAttribute>();
            var key = option?.Key ?? SnakeCase(field.Name);
            criteria[key] = option?.Description;
            byKey[key] = (T)field.GetValue(null)!;
        }
        return (criteria, byKey);
    }

    private static string SnakeCase(string name)
    {
        var b = new StringBuilder();
        for (var i = 0; i < name.Length; i++)
        {
            if (char.IsUpper(name[i]) && i > 0) b.Append('_');
            b.Append(char.ToLowerInvariant(name[i]));
        }
        return b.ToString();
    }
}

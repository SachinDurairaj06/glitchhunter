// GlitchHunter bridge for Unity (2021.3 LTS or newer, built-in or URP/HDRP).
//
// Lets GlitchHunter drive a standalone build one fixed time step at a time: reload the scene,
// set the camera, press keys, advance frames, grab the picture, ray-cast a pixel. Without
// `--gh-port <port>` on the command line the component removes itself, so it can stay in
// shipping builds.
//
// Setup:
//   1. Put this file in Assets/GlitchHunter/ and add the component to a GameObject in the first
//      scene (it survives scene loads).
//   2. Read keyboard input through GlitchHunterBridge.GetKey / GetKeyDown in gameplay code (or
//      enable the Input System block below), so scripted key presses reach the game.
//   3. Build a standalone player and describe how to launch it in glitchhunter.json:
//        "driver": {"type": "bridge",
//                   "launch": ["{dir}/Build/Game.exe", "-screen-width", "{width}", "-screen-height",
//                              "{height}", "-screen-fullscreen", "0", "--gh-port", "{port}"]}
//
// Protocol: one JSON object per line over TCP, request -> reply (docs/ENGINES.md).
// Status: reference implementation; it follows the protocol the Godot bridge is tested with,
// but it has not been run in a Unity project yet.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GlitchHunterBridge : MonoBehaviour
{
    public const int ApiVersion = 1;
    public const float FrameTime = 1f / 30f;

    static readonly HashSet<KeyCode> Held = new HashSet<KeyCode>();
    static readonly HashSet<KeyCode> PressedThisFrame = new HashSet<KeyCode>();
    static GlitchHunterBridge _instance;

    /// <summary>True when GlitchHunter is driving this build.</summary>
    public static bool IsDriven { get; private set; }

    /// <summary>Use instead of Input.GetKey so scripted key presses reach the game.</summary>
    public static bool GetKey(KeyCode key) => IsDriven ? Held.Contains(key) : Input.GetKey(key);

    /// <summary>Use instead of Input.GetKeyDown. A scripted press shows in the first frame that
    /// advances after it, never in a paused frame, so every run sees it with the same delta time.</summary>
    public static bool GetKeyDown(KeyCode key) =>
        IsDriven ? Time.timeScale > 0f && PressedThisFrame.Contains(key) : Input.GetKeyDown(key);

    TcpClient _client;
    StreamReader _reader;
    StreamWriter _writer;
    int _frame;
    int _remaining;
    string _replyAfterAdvance;
    bool _resetting;
    readonly List<Dictionary<string, object>> _errors = new List<Dictionary<string, object>>();

    void Awake()
    {
        int port = PortFromArgs();
        if (port <= 0 || _instance != null)
        {
            // Not driven, or the scene was reloaded by a reset: the copy in the reloaded scene must
            // not open a second connection. Only the component goes; its GameObject may hold others.
            Destroy(this);
            return;
        }
        _instance = this;
        DontDestroyOnLoad(gameObject);
        IsDriven = true;
        Application.runInBackground = true;
        QualitySettings.vSyncCount = 0;
        Time.captureDeltaTime = FrameTime; // every frame advances exactly one fixed step
        Time.fixedDeltaTime = 1f / 60f;
        Time.timeScale = 0f; // paused until GlitchHunter says advance
        Application.logMessageReceived += OnLog;
        SceneManager.sceneLoaded += (scene, mode) => _resetting = false;
        _client = new TcpClient("127.0.0.1", port) { NoDelay = true };
        var stream = _client.GetStream();
        _reader = new StreamReader(stream, new UTF8Encoding(false));
        _writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true, NewLine = "\n" };
        StartCoroutine(Loop());
    }

    static int PortFromArgs()
    {
        var args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--gh-port" && i + 1 < args.Length && int.TryParse(args[i + 1], out int p)) return p;
            if (args[i].StartsWith("--gh-port=") && int.TryParse(args[i].Substring(10), out int q)) return q;
        }
        return 0;
    }

    void OnLog(string message, string stack, LogType type)
    {
        if (type == LogType.Exception || type == LogType.Error)
            _errors.Add(new Dictionary<string, object> { { "type", type.ToString().ToLower() }, { "message", message }, { "frame", _frame } });
    }

    IEnumerator Loop()
    {
        var endOfFrame = new WaitForEndOfFrame();
        while (true)
        {
            yield return endOfFrame; // the frame has been rendered
            if (_remaining > 0)
            {
                PressedThisFrame.Clear(); // this advanced frame delivered the presses
                _remaining--;
                _frame++;
                if (_remaining == 0)
                {
                    Time.timeScale = 0f;
                    Send(_replyAfterAdvance);
                }
                continue;
            }
            if (_resetting) continue;
            while (_client.Connected && _client.Available > 0 && _remaining == 0 && !_resetting)
            {
                string line = _reader.ReadLine();
                if (!string.IsNullOrWhiteSpace(line)) Handle(MiniJson.Parse(line) as Dictionary<string, object>);
            }
            if (!_client.Connected) Application.Quit();
        }
    }

    void Send(object reply) => _writer.WriteLine(reply is string s ? s : MiniJson.Serialize(reply));

    static Dictionary<string, object> Ok(params (string, object)[] fields)
    {
        var d = new Dictionary<string, object> { { "ok", true } };
        foreach (var (k, v) in fields) d[k] = v;
        return d;
    }

    void Handle(Dictionary<string, object> msg)
    {
        if (msg == null) { Send(new Dictionary<string, object> { { "ok", false }, { "error", "bad request" } }); return; }
        string cmd = msg.TryGetValue("cmd", out var c) ? c as string : "";
        switch (cmd)
        {
            case "hello":
                Send(Ok(("api", ApiVersion), ("engine", "unity " + Application.unityVersion),
                        ("viewport", new List<object> { Screen.width, Screen.height })));
                break;
            case "reset":
                UnityEngine.Random.InitState(Convert.ToInt32(msg.TryGetValue("seed", out var s) ? s : 1));
                _frame = 0;
                _errors.Clear();
                Held.Clear();
                PressedThisFrame.Clear();
                _resetting = true;
                StartCoroutine(Reload());
                break;
            case "set_camera":
                Send(SetCamera(msg));
                break;
            case "key":
                var code = ToKeyCode(msg.TryGetValue("code", out var k) ? k as string : "");
                bool down = !msg.TryGetValue("down", out var d) || Convert.ToBoolean(d);
                if (code != KeyCode.None)
                {
                    if (down) { if (Held.Add(code)) PressedThisFrame.Add(code); }
                    else Held.Remove(code);
                }
                Send(Ok());
                break;
            case "advance":
                int n = Math.Max(1, Convert.ToInt32(msg.TryGetValue("frames", out var f) ? f : 1));
                _remaining = n;
                _replyAfterAdvance = MiniJson.Serialize(Ok(("frame", _frame + n)));
                Time.timeScale = 1f;
                break;
            case "grab":
                var tex = ScreenCapture.CaptureScreenshotAsTexture();
                string png = Convert.ToBase64String(tex.EncodeToPNG());
                Send(Ok(("png", png), ("size", new List<object> { tex.width, tex.height })));
                Destroy(tex);
                break;
            case "state":
                Send(Ok(("frame", _frame), ("alive", true), ("errors", _errors), ("camera", CameraSpec())));
                break;
            case "pick":
                Send(Pick(Convert.ToSingle(msg["x"], CultureInfo.InvariantCulture), Convert.ToSingle(msg["y"], CultureInfo.InvariantCulture)));
                break;
            case "quit":
                Send(Ok());
                Application.Quit();
                break;
            default:
                Send(new Dictionary<string, object> { { "ok", false }, { "error", "unknown command " + cmd } });
                break;
        }
    }

    IEnumerator Reload()
    {
        var op = SceneManager.LoadSceneAsync(SceneManager.GetActiveScene().buildIndex);
        while (!op.isDone) yield return null;
        yield return null; // let Awake/Start run
        _resetting = false;
        Send(Ok());
    }

    static Vector3 Vec(object o)
    {
        var l = (List<object>)o;
        return new Vector3(Convert.ToSingle(l[0], CultureInfo.InvariantCulture), Convert.ToSingle(l[1], CultureInfo.InvariantCulture),
                           Convert.ToSingle(l[2], CultureInfo.InvariantCulture));
    }

    static Dictionary<string, object> SetCamera(Dictionary<string, object> msg)
    {
        var cam = Camera.main;
        if (cam == null) return new Dictionary<string, object> { { "ok", false }, { "error", "no main camera" } };
        if (msg.TryGetValue("pos", out var p)) cam.transform.position = Vec(p);
        if (msg.TryGetValue("target", out var t)) cam.transform.LookAt(Vec(t), Vector3.up);
        if (msg.TryGetValue("fov", out var fov)) cam.fieldOfView = Convert.ToSingle(fov, CultureInfo.InvariantCulture);
        return Ok(("camera", CameraSpec()));
    }

    static Dictionary<string, object> CameraSpec()
    {
        var cam = Camera.main;
        if (cam == null) return new Dictionary<string, object>();
        var pos = cam.transform.position;
        var target = pos + cam.transform.forward * 10f;
        return new Dictionary<string, object>
        {
            { "pos", new List<object> { pos.x, pos.y, pos.z } },
            { "target", new List<object> { target.x, target.y, target.z } },
            { "fov", cam.fieldOfView },
        };
    }

    static Dictionary<string, object> Pick(float x, float y)
    {
        var cam = Camera.main;
        if (cam == null) return Ok(("hit", false));
        var ray = cam.ScreenPointToRay(new Vector3(x, Screen.height - y, 0)); // GlitchHunter's y grows downward
        var hits = Physics.RaycastAll(ray, 1000f);
        if (hits.Length == 0) return Ok(("hit", false));
        Array.Sort(hits, (a, b) => a.distance.CompareTo(b.distance));
        var first = hits[0];
        var coplanar = new List<object>();
        foreach (var h in hits)
            if (Mathf.Abs(h.distance - first.distance) < 0.005f && !coplanar.Contains(h.collider.name)) coplanar.Add(h.collider.name);
        return Ok(("hit", true), ("point", new List<object> { first.point.x, first.point.y, first.point.z }),
                  ("object", first.collider.name), ("distance", first.distance), ("coplanar", coplanar),
                  ("camera", CameraSpec()));
    }

    // Browser KeyboardEvent.code names, so one test plan works for web and native builds.
    static KeyCode ToKeyCode(string code)
    {
        if (string.IsNullOrEmpty(code)) return KeyCode.None;
        if (code.StartsWith("Key") && code.Length == 4 && Enum.TryParse(code.Substring(3), out KeyCode letter)) return letter;
        if (code.StartsWith("Digit") && code.Length == 6) return KeyCode.Alpha0 + (code[5] - '0');
        switch (code)
        {
            case "Space": return KeyCode.Space;
            case "Enter": return KeyCode.Return;
            case "Escape": return KeyCode.Escape;
            case "Tab": return KeyCode.Tab;
            case "ShiftLeft": return KeyCode.LeftShift;
            case "ControlLeft": return KeyCode.LeftControl;
            case "ArrowUp": return KeyCode.UpArrow;
            case "ArrowDown": return KeyCode.DownArrow;
            case "ArrowLeft": return KeyCode.LeftArrow;
            case "ArrowRight": return KeyCode.RightArrow;
            default: return KeyCode.None;
        }
    }

    // Minimal JSON (objects, arrays, strings, numbers, booleans, null) so the bridge has no dependencies.
    static class MiniJson
    {
        public static object Parse(string json)
        {
            int i = 0;
            return Value(json, ref i);
        }

        static void Ws(string s, ref int i) { while (i < s.Length && char.IsWhiteSpace(s[i])) i++; }

        static object Value(string s, ref int i)
        {
            Ws(s, ref i);
            if (i >= s.Length) return null;
            char ch = s[i];
            if (ch == '{')
            {
                var d = new Dictionary<string, object>();
                i++;
                Ws(s, ref i);
                if (s[i] == '}') { i++; return d; }
                while (true)
                {
                    Ws(s, ref i);
                    string key = (string)Value(s, ref i);
                    Ws(s, ref i);
                    i++; // :
                    d[key] = Value(s, ref i);
                    Ws(s, ref i);
                    if (s[i++] == '}') return d;
                }
            }
            if (ch == '[')
            {
                var l = new List<object>();
                i++;
                Ws(s, ref i);
                if (s[i] == ']') { i++; return l; }
                while (true)
                {
                    l.Add(Value(s, ref i));
                    Ws(s, ref i);
                    if (s[i++] == ']') return l;
                }
            }
            if (ch == '"')
            {
                var sb = new StringBuilder();
                i++;
                while (s[i] != '"')
                {
                    if (s[i] == '\\')
                    {
                        i++;
                        char e = s[i];
                        if (e == 'n') sb.Append('\n');
                        else if (e == 't') sb.Append('\t');
                        else if (e == 'u') { sb.Append((char)Convert.ToInt32(s.Substring(i + 1, 4), 16)); i += 4; }
                        else sb.Append(e);
                    }
                    else sb.Append(s[i]);
                    i++;
                }
                i++;
                return sb.ToString();
            }
            if (s.Substring(i).StartsWith("true")) { i += 4; return true; }
            if (s.Substring(i).StartsWith("false")) { i += 5; return false; }
            if (s.Substring(i).StartsWith("null")) { i += 4; return null; }
            int start = i;
            while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
            return double.Parse(s.Substring(start, i - start), CultureInfo.InvariantCulture);
        }

        public static string Serialize(object o)
        {
            var sb = new StringBuilder();
            Write(sb, o);
            return sb.ToString();
        }

        static void Write(StringBuilder sb, object o)
        {
            switch (o)
            {
                case null: sb.Append("null"); break;
                case string s:
                    sb.Append('"');
                    foreach (char ch in s)
                    {
                        if (ch == '"' || ch == '\\') sb.Append('\\').Append(ch);
                        else if (ch == '\n') sb.Append("\\n");
                        else if (ch < ' ') sb.Append("\\u").Append(((int)ch).ToString("x4"));
                        else sb.Append(ch);
                    }
                    sb.Append('"');
                    break;
                case bool b: sb.Append(b ? "true" : "false"); break;
                case IDictionary<string, object> d:
                    sb.Append('{');
                    bool first = true;
                    foreach (var kv in d)
                    {
                        if (!first) sb.Append(',');
                        first = false;
                        Write(sb, kv.Key);
                        sb.Append(':');
                        Write(sb, kv.Value);
                    }
                    sb.Append('}');
                    break;
                case IEnumerable list when !(o is string):
                    sb.Append('[');
                    bool firstItem = true;
                    foreach (var item in list)
                    {
                        if (!firstItem) sb.Append(',');
                        firstItem = false;
                        Write(sb, item);
                    }
                    sb.Append(']');
                    break;
                case float f: sb.Append(f.ToString("R", CultureInfo.InvariantCulture)); break;
                case double db: sb.Append(db.ToString("R", CultureInfo.InvariantCulture)); break;
                default: sb.Append(Convert.ToString(o, CultureInfo.InvariantCulture)); break;
            }
        }
    }
}

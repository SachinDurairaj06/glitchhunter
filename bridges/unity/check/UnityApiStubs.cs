// Compile-only stand-ins for the parts of the UnityEngine and UnityEditor APIs the bridge uses,
// with Unity's signatures (2021.3 LTS). They let `dotnet build` in this folder type-check
// GlitchHunterBridge.cs and Editor/GlitchHunterBuild.cs with the C# version Unity compiles with,
// on a machine without Unity. They do nothing at run time and are never copied into a project.

using System;

namespace UnityEngine
{
    public class Object
    {
        public string name { get; set; }
        public static void Destroy(Object obj) { }
        public static void DontDestroyOnLoad(Object target) { }
        public static bool operator ==(Object a, Object b) => ReferenceEquals(a, b);
        public static bool operator !=(Object a, Object b) => !ReferenceEquals(a, b);
        public override bool Equals(object other) => ReferenceEquals(this, other);
        public override int GetHashCode() => 0;
    }

    public class Component : Object
    {
        public GameObject gameObject => null;
        public Transform transform => null;
    }

    public class Behaviour : Component { public bool enabled { get; set; } }

    public class MonoBehaviour : Behaviour
    {
        public Coroutine StartCoroutine(System.Collections.IEnumerator routine) => null;
    }

    public sealed class GameObject : Object { }
    public sealed class Coroutine : YieldInstruction { }
    public class YieldInstruction { }
    public sealed class WaitForEndOfFrame : YieldInstruction { }
    public class AsyncOperation : YieldInstruction { public bool isDone => true; }

    public class Transform : Component
    {
        public Vector3 position { get; set; }
        public Vector3 forward => default;
        public void LookAt(Vector3 worldPosition, Vector3 worldUp) { }
    }

    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 up => new Vector3(0, 1, 0);
        public static Vector3 operator +(Vector3 a, Vector3 b) => new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static Vector3 operator *(Vector3 a, float d) => new Vector3(a.x * d, a.y * d, a.z * d);
    }

    public struct Ray { }

    public struct RaycastHit
    {
        public float distance => 0f;
        public Vector3 point => default;
        public Collider collider => null;
    }

    public class Collider : Component { }

    public sealed class Camera : Behaviour
    {
        public static Camera main => null;
        public float fieldOfView { get; set; }
        public Ray ScreenPointToRay(Vector3 pos) => default;
    }

    public static class Physics { public static RaycastHit[] RaycastAll(Ray ray, float maxDistance) => new RaycastHit[0]; }
    public static class Mathf { public static float Abs(float f) => Math.Abs(f); }
    public static class Screen { public static int width => 0; public static int height => 0; }

    public class Texture : Object { public virtual int width => 0; public virtual int height => 0; }
    public sealed class Texture2D : Texture { }
    public static class ImageConversion { public static byte[] EncodeToPNG(this Texture2D tex) => new byte[0]; }
    public static class ScreenCapture { public static Texture2D CaptureScreenshotAsTexture() => null; }

    public static class Time
    {
        public static float timeScale { get; set; }
        public static float captureDeltaTime { get; set; }
        public static float fixedDeltaTime { get; set; }
    }

    public static class QualitySettings { public static int vSyncCount { get; set; } }

    public static class Input
    {
        public static bool GetKey(KeyCode key) => false;
        public static bool GetKeyDown(KeyCode key) => false;
    }

    public enum KeyCode
    {
        None = 0, Backspace = 8, Tab = 9, Return = 13, Escape = 27, Space = 32,
        Alpha0 = 48, Alpha1, Alpha2, Alpha3, Alpha4, Alpha5, Alpha6, Alpha7, Alpha8, Alpha9,
        A = 97, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        UpArrow = 273, DownArrow = 274, RightArrow = 275, LeftArrow = 276,
        RightShift = 303, LeftShift = 304, RightControl = 305, LeftControl = 306,
    }

    public enum LogType { Error = 0, Assert = 1, Warning = 2, Log = 3, Exception = 4 }

    public static class Application
    {
        public delegate void LogCallback(string condition, string stackTrace, LogType type);
        public static event LogCallback logMessageReceived { add { } remove { } }
        public static bool runInBackground { get; set; }
        public static string unityVersion => "2021.3.0f1";
        public static void Quit() { }
    }

    public static class Random { public static void InitState(int seed) { } }
    public static class Debug { public static void LogError(object message) { } }
}

namespace UnityEngine.Events
{
    public delegate void UnityAction<T0, T1>(T0 arg0, T1 arg1);
}

namespace UnityEngine.SceneManagement
{
    public struct Scene { public int buildIndex => 0; }
    public enum LoadSceneMode { Single = 0, Additive = 1 }

    public static class SceneManager
    {
        public static event UnityEngine.Events.UnityAction<Scene, LoadSceneMode> sceneLoaded { add { } remove { } }
        public static AsyncOperation LoadSceneAsync(int sceneBuildIndex) => null;
        public static Scene GetActiveScene() => default;
    }
}

namespace UnityEditor
{
    public class EditorBuildSettingsScene { public bool enabled { get; set; } public string path { get; set; } }
    public class EditorBuildSettings { public static EditorBuildSettingsScene[] scenes { get; set; } }
    public enum BuildTarget { StandaloneOSX = 2, StandaloneWindows = 5, StandaloneWindows64 = 19, StandaloneLinux64 = 24 }
    [Flags] public enum BuildOptions { None = 0, Development = 1 }

    public struct BuildPlayerOptions
    {
        public string[] scenes { get; set; }
        public string locationPathName { get; set; }
        public BuildTarget target { get; set; }
        public BuildOptions options { get; set; }
    }

    public class BuildPipeline { public static Build.Reporting.BuildReport BuildPlayer(BuildPlayerOptions options) => null; }
    public sealed class EditorApplication { public static void Exit(int returnValue) { } }
}

namespace UnityEditor.Build.Reporting
{
    public enum BuildResult { Unknown = 0, Succeeded = 1, Failed = 2, Cancelled = 3 }
    public struct BuildSummary { public BuildResult result => default; public int totalErrors => 0; }
    public sealed class BuildReport { public BuildSummary summary => default; }
}

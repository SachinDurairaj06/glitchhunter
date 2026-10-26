// Batch-mode player build for GlitchHunter's driver.build step (one build per commit it tests):
//
//   Unity -batchmode -quit -projectPath <dir> -executeMethod GlitchHunterBuild.Build
//
// Builds the scenes enabled in File > Build Settings to Build/Game.exe (Windows 64-bit; change
// Target and the file name for macOS or Linux players). A failed build exits with code 1, which
// GlitchHunter treats like a commit that does not compile: bisect skips it.

#if UNITY_EDITOR
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class GlitchHunterBuild
{
    const BuildTarget Target = BuildTarget.StandaloneWindows64;
    const string Output = "Build/Game.exe";

    public static void Build()
    {
        var options = new BuildPlayerOptions
        {
            scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray(),
            locationPathName = Output,
            target = Target,
            options = BuildOptions.None,
        };
        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result != BuildResult.Succeeded)
        {
            Debug.LogError($"GlitchHunter build failed: {report.summary.result}, {report.summary.totalErrors} error(s)");
            EditorApplication.Exit(1);
        }
    }
}
#endif

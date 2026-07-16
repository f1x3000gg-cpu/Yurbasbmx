#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace HamsterFlip.Editor
{
    public static class AndroidBuilder
    {
        private const string ScenePath = "Assets/Generated/HamsterFlip.unity";

        [MenuItem("Hamster Flip/Generate Scene")]
        public static void GenerateScene()
        {
            Directory.CreateDirectory("Assets/Generated");
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var bootstrap = new GameObject("Hamster Flip Bootstrap");
            bootstrap.AddComponent<HamsterFlip.HamsterBootstrap>();
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        public static void BuildArm64FromCommandLine()
        {
            GenerateScene();
            if (!EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android))
                throw new Exception("Android Build Support is missing.");

            PlayerSettings.companyName = "Independent Test";
            PlayerSettings.productName = "Hamster Flip Test";
            PlayerSettings.applicationIdentifier = "com.independent.hamsterfliptest";
            PlayerSettings.bundleVersion = "0.1.0";
            PlayerSettings.Android.bundleVersionCode = 1;
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
            PlayerSettings.allowedAutorotateToPortrait = false;
            PlayerSettings.allowedAutorotateToPortraitUpsideDown = false;
            PlayerSettings.allowedAutorotateToLandscapeLeft = true;
            PlayerSettings.allowedAutorotateToLandscapeRight = true;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.SetIl2CppCompilerConfiguration(BuildTargetGroup.Android, Il2CppCompilerConfiguration.Release);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel26;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            PlayerSettings.Android.optimizedFramePacing = true;
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });
            EditorUserBuildSettings.androidBuildSystem = AndroidBuildSystem.Gradle;
            EditorUserBuildSettings.buildAppBundle = false;

            string directory = Path.GetFullPath("Build/Android");
            Directory.CreateDirectory(directory);
            string output = Path.Combine(directory, "HamsterFlip_ARM64.apk");
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = output,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception("Build failed: " + report.summary.result + ", errors=" + report.summary.totalErrors);
            Debug.Log("APK built: " + output);
        }
    }
}
#endif

#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace BMX.Editor
{
    public static class AndroidBuilder
    {
        private const string ScenePath = "Assets/Generated/BMXMain.unity";

        [MenuItem("BMX/Generate Main Scene")]
        public static void GenerateMainScene()
        {
            Directory.CreateDirectory("Assets/Generated");
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject bootstrap = new GameObject("BMX Bootstrap");
            bootstrap.AddComponent<BMX.SpaceFeel.BMXBootstrap>();
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new EditorBuildSettingsScene[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("Generated " + ScenePath);
        }

        [MenuItem("BMX/Build Android ARM64 IL2CPP")]
        public static void BuildArm64()
        {
            BuildInternal(false);
        }

        public static void BuildArm64FromCommandLine()
        {
            BuildInternal(true);
        }

        private static void BuildInternal(bool batchMode)
        {
            GenerateMainScene();
            if (!EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android))
                throw new Exception("Android build target could not be activated. Install Android Build Support, SDK, NDK and OpenJDK for Unity 2022.3.");
            ConfigurePlayer();

            string outputDirectory = Path.GetFullPath("Build/Android");
            Directory.CreateDirectory(outputDirectory);
            string output = Path.Combine(outputDirectory, "BMX_SpaceFeel_ARM64.apk");
            BuildPlayerOptions options = new BuildPlayerOptions();
            options.scenes = new string[] { ScenePath };
            options.locationPathName = output;
            options.target = BuildTarget.Android;
            options.targetGroup = BuildTargetGroup.Android;
            options.options = batchMode ? BuildOptions.None : BuildOptions.Development | BuildOptions.AllowDebugging;
            BuildReport report = BuildPipeline.BuildPlayer(options);
            BuildSummary summary = report.summary;
            if (summary.result != BuildResult.Succeeded)
                throw new Exception("Android build failed: " + summary.result + ", errors=" + summary.totalErrors);
            Debug.Log("ARM64 APK built: " + output + " (" + summary.totalSize + " bytes)");
        }

        private static void ConfigurePlayer()
        {
            PlayerSettings.companyName = "Independent Clean Room Project";
            PlayerSettings.productName = "BMX SpaceFeel";
            PlayerSettings.applicationIdentifier = "com.cleanroom.bmxspacefeel";
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
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new GraphicsDeviceType[] { GraphicsDeviceType.Vulkan, GraphicsDeviceType.OpenGLES3 });
            EditorUserBuildSettings.androidBuildSystem = AndroidBuildSystem.Gradle;
            EditorUserBuildSettings.buildAppBundle = false;
        }
    }
}
#endif

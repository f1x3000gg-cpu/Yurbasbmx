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
        private const string ScenePath = "Assets/Generated/HamsterTale.unity";

        [MenuItem("Hamster Tale/Generate Scene")]
        public static void GenerateScene()
        {
            Directory.CreateDirectory("Assets/Generated");
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject bootstrap = new GameObject("Hamster Tale Bootstrap");
            bootstrap.AddComponent<HamsterFlip.HamsterTaleBootstrap>();
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        // Keep this exact public method name because Unity Build Automation already calls it.
        public static void PreExport()
        {
            ConfigureAndroidPlayer();
            GenerateScene();
            Debug.Log("Hamster Tale: Unity Build Automation pre-export setup completed.");
        }

        private static void ConfigureAndroidPlayer()
        {
            PlayerSettings.companyName = "DippiX Games";
            PlayerSettings.productName = "Hamster Tale - Last Seed";
            PlayerSettings.applicationIdentifier = "com.f1x3000gg.hamsterflip";
            PlayerSettings.bundleVersion = "1.0.0";
            PlayerSettings.Android.bundleVersionCode = 2;
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
        }

        public static void BuildArm64FromCommandLine()
        {
            if (!EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android))
                throw new Exception("Android Build Support is missing.");

            ConfigureAndroidPlayer();
            GenerateScene();

            string directory = Path.GetFullPath("Build/Android");
            Directory.CreateDirectory(directory);
            string output = Path.Combine(directory, "HamsterTale_LastSeed_ARM64.apk");
            BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
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

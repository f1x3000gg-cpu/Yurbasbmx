# Hamster Tale: The Last Seed — Unity Android ARM64

An original 2D story RPG for Android, built entirely through Unity Build Automation.

## Chapter 1

- Pixel-art hamster hero named Pix
- Short opening story and explorable pantry room
- NPC dialogue and environmental interactions
- Original battle against the Jar Guardian
- Touch movement and direct drag control during enemy attacks
- Talk, Spare, Snack, and Attack actions
- Three projectile patterns
- Peaceful and dark endings saved on the device

## Android build

- Unity 2022.3.62f2
- IL2CPP Release
- ARM64 only
- OpenGLES3
- APK output
- Minimum Android API 26

Unity Build Automation keeps using the existing pre-export hook:

`HamsterFlip.Editor.AndroidBuilder.PreExport`

The new APK replaces the old Hamster Flip test because it keeps the same package identifier and uses a higher version code.

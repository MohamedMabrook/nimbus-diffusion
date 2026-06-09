; Nimbus Diffusion — Inno Setup installer script
; Copyright (C) 2026 Mohamed Mabrok — GPL v3

#define AppName      "Nimbus Diffusion"
#define AppVersion   "2.3"
#define AppAuthor    "Mohamed Mabrok"
#define AppURL       "https://github.com/mohamedmabrok/nimbus-diffusion"
#define OFXPluginDir "C:\Program Files\Common Files\OFX\Plugins"
#define BundleName   "NimbusDiffusor.ofx.bundle"
#define BuildDir     "..\ofx\build\NimbusDiffusor.ofx.bundle"

[Setup]
AppId={{B3F1A2C4-9D7E-4F6B-8A3C-1E5D2F7G9H4J}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppAuthor}
AppPublisherURL={#AppURL}
AppCopyright=Copyright (C) 2026 {#AppAuthor}

; Install into OFX plugins folder
DefaultDirName={#OFXPluginDir}\{#BundleName}
DirExistsWarning=no
DisableDirPage=yes
DisableProgramGroupPage=yes

; Require admin (needed to write to Program Files)
PrivilegesRequired=admin

OutputDir=.
OutputBaseFilename=NimbusDiffusion-{#AppVersion}-Windows-Setup
Compression=lzma2/ultra64
SolidCompression=yes

; Wizard appearance
WizardStyle=modern
WizardImageFile=..\installer_sidebar.bmp
WizardSmallImageFile=..\installer_small.bmp

; No uninstall needed for OFX plugins (optional but nice)
Uninstallable=yes
UninstallDisplayName={#AppName}
UninstallDisplayIcon={#OFXPluginDir}\{#BundleName}\Contents\Win64\NimbusDiffusor.ofx

; Minimum Windows version: Windows 10
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; OFX binary
Source: "{#BuildDir}\Contents\Win64\NimbusDiffusor.ofx"; \
  DestDir: "{app}\Contents\Win64"; \
  Flags: ignoreversion

; Plugin icon
Source: "{#BuildDir}\Contents\Resources\NimbusDiffusion.png"; \
  DestDir: "{app}\Contents\Resources"; \
  Flags: ignoreversion

; License
Source: "..\LICENSE"; \
  DestDir: "{app}"; \
  Flags: ignoreversion

; README
Source: "..\README.md"; \
  DestDir: "{app}"; \
  Flags: ignoreversion isreadme

[Dirs]
Name: "{app}\Contents\Win64"
Name: "{app}\Contents\Resources"

[Run]
; Remind user to clear OFX cache
Filename: "{sys}\notepad.exe"; \
  Parameters: "{app}\README.md"; \
  Description: "View README"; \
  Flags: postinstall skipifsilent unchecked

[Messages]
WelcomeLabel1=Welcome to Nimbus Diffusion {#AppVersion}
WelcomeLabel2=This will install Nimbus Diffusion — a free optical diffusion plugin for DaVinci Resolve, by Mohamed Mabrok.%n%nPlease close DaVinci Resolve before continuing.
FinishedLabel=Nimbus Diffusion {#AppVersion} has been installed.%n%nIf DaVinci Resolve was open, restart it now. If controls look wrong, clear the OFX cache:%n  Preferences → System → Memory and GPU → Clear OFX cache

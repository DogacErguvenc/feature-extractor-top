#define MyAppName "Terazi Backend"
#define MyAppPublisher "Terazi AI"
#define MyTaskName "terazi-ai-backend"
#define MyAppExeName "terazi_backend.exe"

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#ifndef SourceDir
  #define SourceDir "..\dist\terazi_backend"
#endif

[Setup]
AppId={{38D7DF80-1C72-4A92-946E-F1680D6E0148}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName=C:\TeraziBackend
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=terazi_backend_setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "autostart"; Description: "Windows acilisinda backend'i otomatik baslat"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Terazi Backend\Backend Baslat"; Filename: "{app}\start_backend.bat"
Name: "{autodesktop}\Terazi Backend"; Filename: "{app}\start_backend.bat"

[Run]
Filename: "{app}\start_backend.bat"; Description: "Backend'i simdi baslat"; Flags: postinstall skipifsilent nowait

[Code]
function RunSchtasks(const Params: string): Boolean;
var
  ResultCode: Integer;
begin
  Log('schtasks ' + Params);
  Result := Exec(ExpandConstant('{sys}\schtasks.exe'), Params, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  if Result then
    Result := (ResultCode = 0);
end;

procedure RegisterStartupTask();
var
  TrValue: string;
  Params: string;
begin
  TrValue := ExpandConstant('{cmd}') + ' /c ' + AddQuotes(ExpandConstant('{app}\start_backend.bat'));
  Params :=
    '/Create /F /RL HIGHEST /SC ONSTART /RU SYSTEM ' +
    '/TN "{#MyTaskName}" /TR ' + AddQuotes(TrValue);

  if not RunSchtasks(Params) then
    MsgBox(
      'Windows acilis gorevi olusturulamadi. Elle ekleyin: {#MyTaskName}',
      mbError,
      MB_OK
    );
end;

procedure UnregisterStartupTask();
begin
  RunSchtasks('/Delete /F /TN "{#MyTaskName}"');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('autostart') then
    RegisterStartupTask();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    UnregisterStartupTask();
end;

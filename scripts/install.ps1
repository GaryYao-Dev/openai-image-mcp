# Requires -Version 5.1
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$uvVersion = "0.11.23"

function Find-UvPath {
    $command = Get-Command -Name "uv" -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $command) {
        return $command.Path
    }

    $installDirectory = $env:UV_INSTALL_DIR
    if ([string]::IsNullOrWhiteSpace($installDirectory)) {
        $installDirectory = Join-Path -Path $HOME -ChildPath ".local\bin"
    }
    $installedPath = Join-Path -Path $installDirectory -ChildPath "uv.exe"
    if (Test-Path -LiteralPath $installedPath -PathType Leaf) {
        return $installedPath
    }

    return $null
}

function Assert-UvWorks {
    param([Parameter(Mandatory = $true)][string]$UvPath)

    & $UvPath --version
    if ($LASTEXITCODE -ne 0) {
        throw "uv was found but could not run: $UvPath"
    }
}

$uvPath = Find-UvPath
if ($null -ne $uvPath) {
    Assert-UvWorks -UvPath $uvPath
    Write-Host "uv is already available: $uvPath"
    return
}

$installerPath = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath "openai-image-mcp-uv-$PID.ps1"
try {
    $download = @{
        Uri = "https://astral.sh/uv/$uvVersion/install.ps1"
        OutFile = $installerPath
    }
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        $download.UseBasicParsing = $true
    }
    Invoke-WebRequest @download

    $powershellPath = (Get-Command -Name "powershell.exe" -CommandType Application -ErrorAction Stop).Path
    & $powershellPath -NoProfile -ExecutionPolicy Bypass -File $installerPath
    if ($LASTEXITCODE -ne 0) {
        throw "The official uv installer exited with code $LASTEXITCODE."
    }
}
finally {
    if (Test-Path -LiteralPath $installerPath) {
        Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
    }
}

$uvPath = Find-UvPath
if ($null -eq $uvPath) {
    throw "uv was installed but is not discoverable. Restart your shell, then run this script again."
}

Assert-UvWorks -UvPath $uvPath
Write-Host "uv installed: $uvPath"
Write-Host "Restart Codex so it inherits the updated PATH, then enable or restart the OpenAI Image MCP."

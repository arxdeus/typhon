# Requires PowerShell 7 and an x64 MSVC developer environment (including Windows SDK).
# Workflow: ilammy/msvc-dev-cmd@v1 (arch: x64), then shell: pwsh,
# run: ./tools/ci/setup-windows.ps1
# Full LLVM archives include llvm-config, headers and static libraries. The LLVM
# NSIS installer is not a substitute for this development distribution.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $IsWindows) { throw 'This setup requires Windows.' }
foreach ($name in @('RUNNER_TEMP', 'GITHUB_ENV', 'GITHUB_PATH')) {
    if (-not [Environment]::GetEnvironmentVariable($name)) { throw "Missing $name" }
}
foreach ($tool in @('cl.exe', 'lib.exe', '7z.exe', 'curl.exe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing $tool. Run ilammy/msvc-dev-cmd with arch: x64 before this step."
    }
}

$root = Join-Path $env:RUNNER_TEMP 'typhon-windows-deps'
New-Item -ItemType Directory -Force -Path $root | Out-Null
function Get-VerifiedArchive {
    param([string]$Name, [string]$Url, [string]$Sha256)
    $archive = Join-Path $root $Name
    if (Test-Path -LiteralPath $archive) {
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -eq $Sha256) { return $archive }
        Remove-Item -LiteralPath $archive -Force
    }
    $partial = "$archive.partial"
    # Use the upstream API's browser_download_url spelling, notably %2B for LLVM.
    & curl.exe --fail --location --retry 5 --retry-all-errors --connect-timeout 30 --max-time 1800 --output $partial $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url (exit $LASTEXITCODE)" }
    $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) { throw "SHA256 mismatch for ${Name}: expected $Sha256, received $actual" }
    Move-Item -LiteralPath $partial -Destination $archive -Force
    return $archive
}
function Set-CiEnvironment {
    param([string]$Name, [string]$Value)
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
    "$Name=$Value" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
}
function Add-CiPath {
    param([string]$Directory)
    $Directory | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
    $env:PATH = "$Directory;$env:PATH"
}
function Assert-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required dependency file missing: $Path" }
}

# Asset names and SHA256 digests verified against the respective GitHub release APIs.
$llvmArchive = Get-VerifiedArchive 'llvm-20.1.8-windows-x64.tar.xz' `
    'https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.8/clang%2Bllvm-20.1.8-x86_64-pc-windows-msvc.tar.xz' `
    'f229769f11d6a6edc8ada599c0cda964b7dee6ab1a08c6cf9dd7f513e85b107f'
$blasArchive = Get-VerifiedArchive 'OpenBLAS-0.3.30-x64.zip' `
    'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30-x64.zip' `
    '8b04387766efc05c627e26d24797ec0d4ed4c105ec14fa7400aa84a02db22b66'

# Extract into private directories, without assuming the LLVM archive's top-level name.
$llvmExtract = Join-Path $root 'llvm-20.1.8'
$blas = Join-Path $root 'openblas-0.3.30'
foreach ($directory in @($llvmExtract, $blas)) {
    if (Test-Path -LiteralPath $directory) { Remove-Item -LiteralPath $directory -Recurse -Force }
    New-Item -ItemType Directory -Path $directory | Out-Null
}
# Windows' bundled bsdtar can spend tens of minutes decompressing this archive.
# Stream 7-Zip's xz output into a second 7-Zip process, without a multi-GB .tar
# intermediate. BaseStream avoids PowerShell decoding/corrupting binary output.
Write-Host 'Extracting LLVM development archive with 7-Zip...'
$sevenZip = (Get-Command 7z.exe).Source
$decodeInfo = [System.Diagnostics.ProcessStartInfo]::new($sevenZip)
$decodeInfo.UseShellExecute = $false
$decodeInfo.RedirectStandardOutput = $true
foreach ($argument in @('x', $llvmArchive, '-so', '-bsp0')) { $decodeInfo.ArgumentList.Add($argument) }
$extractInfo = [System.Diagnostics.ProcessStartInfo]::new($sevenZip)
$extractInfo.UseShellExecute = $false
$extractInfo.RedirectStandardInput = $true
foreach ($argument in @('x', '-si', '-ttar', "-o$llvmExtract", '-y', '-bso0', '-bsp0')) { $extractInfo.ArgumentList.Add($argument) }
$decoder = $null
$extractor = $null
try {
    $extractor = [System.Diagnostics.Process]::Start($extractInfo)
    $decoder = [System.Diagnostics.Process]::Start($decodeInfo)
    $decoder.StandardOutput.BaseStream.CopyTo($extractor.StandardInput.BaseStream)
    $extractor.StandardInput.Close()
    $decoder.WaitForExit()
    $extractor.WaitForExit()
    if ($decoder.ExitCode -ne 0 -or $extractor.ExitCode -ne 0) {
        throw "LLVM extraction failed (xz=$($decoder.ExitCode), tar=$($extractor.ExitCode))"
    }
} finally {
    foreach ($process in @($decoder, $extractor)) {
        if ($null -ne $process) {
            if (-not $process.HasExited) { $process.Kill() }
            $process.Dispose()
        }
    }
}
Write-Host 'LLVM extracted; extracting OpenBLAS...'
Expand-Archive -LiteralPath $blasArchive -DestinationPath $blas
$configs = @(Get-ChildItem -LiteralPath $llvmExtract -Filter llvm-config.exe -Recurse -File)
if ($configs.Count -ne 1) { throw "Expected exactly one llvm-config.exe, found $($configs.Count)" }
$llvm = Split-Path (Split-Path $configs[0].FullName -Parent) -Parent
$config = $configs[0].FullName
foreach ($relative in @('bin/clang.exe', 'bin/clang-cl.exe', 'bin/lld-link.exe', 'include/llvm-c/Core.h', 'lib/LLVMCore.lib')) {
    Assert-File (Join-Path $llvm $relative)
}
foreach ($relative in @('bin/libopenblas.dll', 'lib/libopenblas.lib', 'include/cblas.h')) {
    Assert-File (Join-Path $blas $relative)
}

# The official LLVM Windows archive references libxml2s.lib but does not ship it.
# Build a /MT static libxml2 without optional compression/iconv dependencies.
$xmlLibrary = Join-Path $llvm 'lib/libxml2s.lib'
if (-not (Test-Path -LiteralPath $xmlLibrary)) {
    Write-Host 'Building the missing LLVM libxml2 static dependency with vcpkg...'
    $vcpkg = (Get-Command vcpkg.exe -ErrorAction Stop).Source
    $installed = Join-Path $root 'vcpkg-installed'
    & $vcpkg install 'libxml2[core]:x64-windows-static' --classic "--x-install-root=$installed" --disable-metrics
    if ($LASTEXITCODE -ne 0) { throw "libxml2 build failed (exit $LASTEXITCODE)" }
    $xmlLibraries = @(Get-ChildItem -LiteralPath (Join-Path $installed 'x64-windows-static/lib') -Filter '*xml2*.lib' -File)
    if ($xmlLibraries.Count -ne 1) { throw "Expected one static libxml2 library, found $($xmlLibraries.Count)" }
    Copy-Item -LiteralPath $xmlLibraries[0].FullName -Destination $xmlLibrary
    Copy-Item -LiteralPath (Join-Path $installed 'x64-windows-static/share/libxml2/copyright') -Destination (Join-Path $llvm 'libxml2-copyright.txt')
}

Set-CiEnvironment 'LLVM_SYS_201_PREFIX' $llvm
Set-CiEnvironment 'OPENBLAS_ROOT' $blas
Set-CiEnvironment 'TYPHON_OPENBLAS_DIR' $blas
# Preserve the developer command prompt's CRT and Windows SDK library search paths.
Set-CiEnvironment 'LIB' "$(Join-Path $blas 'lib');$(Join-Path $llvm 'lib');$env:LIB"
# LLVM's official static libraries use /MT (MT_StaticRelease). Compile target
# crates and cc-built shims with the same CRT. Use an explicit --target in Cargo
# so host proc macros/build scripts do not inherit the target's crt-static flag.
Set-CiEnvironment 'ARTIFACT_DIR' 'target/x86_64-pc-windows-msvc/release'
Set-CiEnvironment 'CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS' '-C target-feature=+crt-static'
Set-CiEnvironment 'CFLAGS_x86_64_pc_windows_msvc' '-DLARGE_CONFIG -DNO_BLACK_LISTING'
# Keep the required language level explicit for MSVC native dependencies.
$cxxFlags = [Environment]::GetEnvironmentVariable('CXXFLAGS_x86_64_pc_windows_msvc')
Set-CiEnvironment 'CXXFLAGS_x86_64_pc_windows_msvc' "$cxxFlags /std:c++17".Trim()
Add-CiPath (Join-Path $llvm 'bin')
Add-CiPath (Join-Path $blas 'bin')

$version = & $config --version
if ($LASTEXITCODE -ne 0 -or $version.Trim() -ne '20.1.8') { throw "Unexpected LLVM version: $version" }
# llvm-sys 201 defaults to static LLVM and obtains additional dependencies from
# --system-libs. Fail early on missing development libraries, rather than at rustc.
$llvmLibraries = & $config --link-static --libnames
if ($LASTEXITCODE -ne 0) { throw 'llvm-config --link-static --libnames failed' }
foreach ($library in ($llvmLibraries -split '\s+' | Where-Object { $_ })) {
    Assert-File (Join-Path $llvm "lib/$library")
}
$systemLibraries = & $config --link-static --system-libs
if ($LASTEXITCODE -ne 0) { throw 'llvm-config --link-static --system-libs failed' }
foreach ($library in ($systemLibraries -split '\s+' | Where-Object { $_ })) {
    $found = Test-Path -LiteralPath $library -PathType Leaf
    foreach ($directory in ($env:LIB -split ';' | Where-Object { $_ })) {
        if (Test-Path -LiteralPath (Join-Path $directory $library) -PathType Leaf) { $found = $true; break }
    }
    if (-not $found) { throw "LLVM static system dependency not found in LIB: $library" }
}
Write-Host "LLVM $version prefix: $llvm"
Write-Host "LLVM static system dependencies: $systemLibraries"
Write-Host "OpenBLAS LP64 DLL and MSVC import library: $blas"
& (Join-Path $llvm 'bin/clang.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'clang failed to run' }
& (Join-Path $llvm 'bin/lld-link.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'lld-link failed to run' }

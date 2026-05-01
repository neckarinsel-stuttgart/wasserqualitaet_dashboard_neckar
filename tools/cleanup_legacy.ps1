param(
    [switch]$Apply,
    [switch]$ArchiveSkripts,
    [switch]$ArchiveDatenDeprecated,
    [switch]$DeletePapermillOutputs,
    [string]$ArchiveDir = "archive"
)

$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
    param([string]$Start = $PWD.Path)
    $cur = (Resolve-Path $Start).Path
    while ($true) {
        if (Test-Path (Join-Path $cur 'databricks.yml')) { return $cur }
        $parent = Split-Path $cur -Parent
        if ($parent -eq $cur -or [string]::IsNullOrWhiteSpace($parent)) { return (Resolve-Path $Start).Path }
        $cur = $parent
    }
}

$root = Get-RepoRoot
$timestamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$archiveRoot = Join-Path $root $ArchiveDir
$archiveRunDir = Join-Path $archiveRoot $timestamp

$script:actions = @()

function Add-MoveAction {
    param([string]$FromRel, [string]$ToRel)
    $from = Join-Path $root $FromRel
    if (-not (Test-Path $from)) { return }
    $to = Join-Path $archiveRunDir $ToRel
    $script:actions += [pscustomobject]@{ Type='Move'; From=$from; To=$to }
}

function Add-DeleteAction {
    param([string]$TargetRel)
    $target = Join-Path $root $TargetRel
    if (-not (Test-Path $target)) { return }
    $script:actions += [pscustomobject]@{ Type='Delete'; Path=$target }
}

if ($ArchiveSkripts) {
    Add-MoveAction -FromRel 'skripts' -ToRel 'skripts'
}

if ($ArchiveDatenDeprecated) {
    Add-MoveAction -FromRel (Join-Path 'Daten' 'deprecated') -ToRel (Join-Path 'Daten' 'deprecated')
}

if ($DeletePapermillOutputs) {
    $dataGold = Join-Path (Join-Path 'data' 'gold') 'papermill_outputs'
    $skriptsPm = Join-Path 'skripts' 'papermill_outputs'
    Add-DeleteAction -TargetRel $dataGold
    Add-DeleteAction -TargetRel $skriptsPm
}

Write-Host "Repo root: $root"
Write-Host "Planned actions: $($script:actions.Count)"

if ($script:actions.Count -eq 0) {
    Write-Host "Nothing to do. Use flags like -ArchiveSkripts -DeletePapermillOutputs."
    exit 0
}

$script:actions | Format-Table Type,From,To,Path -AutoSize

Write-Host ""
foreach ($a in $script:actions) {
    if ($a.Type -eq 'Move') {
        Write-Host ("MOVE  : {0} -> {1}" -f $a.From, $a.To)
    } elseif ($a.Type -eq 'Delete') {
        Write-Host ("DELETE: {0}" -f $a.Path)
    }
}

if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply to execute." 
    exit 0
}

New-Item -ItemType Directory -Force -Path $archiveRunDir | Out-Null

foreach ($a in $script:actions) {
    if ($a.Type -eq 'Move') {
        New-Item -ItemType Directory -Force -Path (Split-Path $a.To -Parent) | Out-Null
        Write-Host "Moving: $($a.From) -> $($a.To)"
        Move-Item -Force -Path $a.From -Destination $a.To
    } elseif ($a.Type -eq 'Delete') {
        Write-Host "Deleting: $($a.Path)"
        Remove-Item -Recurse -Force -Path $a.Path
    }
}

Write-Host "Done. Archive created at: $archiveRunDir"
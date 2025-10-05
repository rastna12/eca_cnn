param(
    [string]$RunsDir = "runs",
    [string]$FigsDir = "figs",
    [string]$AnalysisDir = "figs/analysis",
    [string]$Device = "cuda",
    [int]$Steps = 2000,
    [string]$Rules = "150,90,30,110",
    [string]$Hs = "1,2,4,8,16,32",
    [string]$Models = "shallow",
    [string]$Depths = "",
    [string]$Seeds = "1",
    [int]$Width = 256,
    [int]$SpacetimeSteps = 256,
    [int]$Hmask = 16,
    [int]$QualH = 32,
    [int]$QualSeed = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Tool {
    param([string]$Cmd)
    Write-Host ">> $Cmd"
    iex $Cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Cmd"
    }
}

# Determine how to run console scripts (prefer uv)
$UseUv = $false
if (Get-Command uv -ErrorAction SilentlyContinue) { $UseUv = $true }

$Runner = if ($UseUv) { "uv run" } else { "python -m" }
Write-Host "Using $Runner to invoke tools."

# Clean previous outputs: delete contents of RunsDir and FigsDir
foreach ($dir in @($RunsDir, $FigsDir)) {
    try {
        if (Test-Path -LiteralPath $dir) {
            Write-Host "Cleaning directory: $dir"
            Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        }
        else {
            Write-Host "Creating directory: $dir"
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    catch {
        Write-Warning "Failed to clean directory: $dir. $_"
    }
}

# 1) Generate base ECA spacetime and parity mask plots (optional but useful)
if ($UseUv) {
    Invoke-Tool "uv run eca-cnn-plots --outdir `"$FigsDir`" --width $Width --steps $SpacetimeSteps --Hmask $Hmask"
}
else {
    Invoke-Tool "python -m eca_cnn.eca_plots --outdir `"$FigsDir`" --width $Width --steps $SpacetimeSteps --Hmask $Hmask"
}

# 2) Run experiment sweeps (trains and saves artifacts)
# Build experiments command; omit --depths when empty to avoid argparse error
$depthsArg = if ([string]::IsNullOrWhiteSpace($Depths)) { "" } else { " --depths `"$Depths`"" }
if ($UseUv) {
    $cmd = "uv run eca-cnn-experiments --rules `"$Rules`" --Hs `"$Hs`" --models `"$Models`"$depthsArg --seeds `"$Seeds`" --steps $Steps --device $Device --runs-dir `"$RunsDir`""
    Invoke-Tool $cmd
}
else {
    $cmd = "python -m eca_cnn.experiments --rules `"$Rules`" --Hs `"$Hs`" --models `"$Models`"$depthsArg --seeds `"$Seeds`" --steps $Steps --device $Device --runs-dir `"$RunsDir`""
    Invoke-Tool $cmd
}

# 3) Aggregate and plot analysis from saved runs
$latexOut = Join-Path $AnalysisDir "metrics_table.tex"
if ($UseUv) {
    Invoke-Tool "uv run eca-cnn-analysis --runs-dir `"$RunsDir`" --outdir `"$AnalysisDir`" --rules `"$Rules`" --Hs `"$Hs`" --qual-H $QualH --qual-seed $QualSeed --latex-table --latex-out `"$latexOut`""
}
else {
    Invoke-Tool "python -m eca_cnn.analysis --runs-dir `"$RunsDir`" --outdir `"$AnalysisDir`" --rules `"$Rules`" --Hs `"$Hs`" --qual-H $QualH --qual-seed $QualSeed --latex-table --latex-out `"$latexOut`""
}

Write-Host "All done." -ForegroundColor Green
Write-Host "Runs saved under: $RunsDir"
Write-Host "Figures saved under: $FigsDir and $AnalysisDir"
Write-Host "LaTeX table saved to: $latexOut"



param([switch]$Download,[string]$Predictions='')
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$taskArgs=@('-m','soundmap.external_validation')
if ($Download) {$taskArgs+='--download'}
if ($Predictions) {$taskArgs+=@('--predictions',$Predictions)}
& .\.venv\Scripts\python.exe @taskArgs
if ($LASTEXITCODE -ne 0) {throw 'External benchmark audit failed.'}

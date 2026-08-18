param(
    [switch]$Live
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$projectRoot = $PSScriptRoot
$python = "C:\Users\ASUS\miniconda3\envs\know\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到 know 环境：$python"
}

Push-Location $projectRoot
try {
    & $python -m unittest knowledge.test.query_process.test_query_completion -v
    & $python evaluation\verify_offline_artifacts.py

    if ($Live) {
        & $python knowledge\test\test_connections.py
        $env:CHUNKS_COLLECTION = "appliance_chunks_v2"
        $env:ITEM_NAME_COLLECTION = "appliance_items_v2"
        $env:ENTITY_NAME_COLLECTION = "appliance_entities_v2"
        $env:QUERY_RETRIEVAL_MODE = "dense"
        & $python evaluation\run_evaluation.py `
            --dataset evaluation\datasets\panasonic_vacuum_retrieval_v1.jsonl `
            --output evaluation\reports\live_smoke_latest.jsonl `
            --limit 1
    }
}
finally {
    Pop-Location
}

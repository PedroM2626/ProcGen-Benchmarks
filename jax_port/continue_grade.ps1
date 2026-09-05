# Vigia da grade: quando o run_grade.py atual terminar, lanca o catch-up
# (suites spr+aux+gnn: retry dos SPR falhos + CURL/CPC/ACL novos + resto
# do GNN), com resume — nada e sobrescrito. Uso:
#   powershell -ExecutionPolicy Bypass -File jax_port/continue_grade.ps1
# Ele proprio roda detached; acompanhe por jax_port/results_grade/watcher.log
$repo = "C:\Users\Acer\Downloads\MLE"
$master = "$repo\jax_port\results_grade\master_full.json"
$log = "$repo\jax_port\results_grade\watcher.log"
"watcher started $(Get-Date -Format 'HH:mm:ss')" | Out-File $log -Append
$calm = 0
while ($true) {
  Start-Sleep -Seconds 180
  $run = wsl -e pgrep -f run_grade.py 2>$null
  if ([string]::IsNullOrWhiteSpace("$run")) { $calm++ } else { $calm = 0 }
  "check $(Get-Date -Format 'HH:mm:ss') calm=$calm" | Out-File $log -Append
  if ($calm -ge 3) {
    "launching catch-up $(Get-Date -Format 'HH:mm:ss')" | Out-File $log -Append
    Start-Process -FilePath "wsl" -ArgumentList @('-e','env','PYTHONPATH=/mnt/c/Users/Acer/Downloads/MLE','XLA_PYTHON_CLIENT_MEM_FRACTION=0.9','/root/procgen-jax/bin/python','/mnt/c/Users/Acer/Downloads/MLE/jax_port/run_grade.py','--suite','spr','aux','gnn','--seeds','42','43','44','45','46','--timesteps','100000','--eval-full','--out-dir','/mnt/c/Users/Acer/Downloads/MLE/jax_port/results_grade','--master','/mnt/c/Users/Acer/Downloads/MLE/jax_port/results_grade/master_full.json') -RedirectStandardOutput "$repo\jax_port\results_grade\grade_aux.log" -RedirectStandardError "$repo\jax_port\results_grade\grade_aux.err.log" -WorkingDirectory $repo
    "catch-up launched" | Out-File $log -Append
    break
  }
}

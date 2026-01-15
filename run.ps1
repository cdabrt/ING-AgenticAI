@@ -1,26 +1,3 @@
# nb - no build
# d - compose down
# f - force remove
param(
    [switch]$nb,
    [switch]$d,
    [switch]$f
)
if (!$d.IsPresent -and !$f.IsPresent) {
    if (!$nb.IsPresent) {
        Write-Host -ForegroundColor Yellow "Running docker compose build!"
        docker compose build
    }
    Write-Host -ForegroundColor Yellow "Running docker compose up!"
    docker compose -f .\docker-compose.local.yml up -d --build
}
else {
    if ($f.IsPresent) {
        Write-Host -ForegroundColor Red "Forcing deletion of containers!"
        docker compose -f .\docker-compose.local.yml down --rmi all -v --remove-orphans
    }
    else {
        Write-Host -ForegroundColor Red "Running docker compose down!"
        docker compose -f .\docker-compose.local.yml down
    }
}
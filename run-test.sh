#!/bin/bash

# nb - no build
# d - compose down
# f - force remove

nb=false
d=false
f=false

while getopts "ndf" opt; do
  case $opt in
    n) nb=true ;;
    d) d=true ;;
    f) f=true ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ "$d" = false ] && [ "$f" = false ]; then
    if [ "$nb" = false ]; then
        echo -e "\033[0;33mRunning docker compose build!\033[0m"
        docker compose build
        echo -e "\033[0;33mRunning docker compose up!\033[0m"
        docker compose -f ./docker-compose.local.test.yml up -d --build
    else
        echo -e "\033[0;33mRunning docker compose up without building!\033[0m"
        docker compose -f ./docker-compose.local.test.yml up -d
    fi
else
    if [ "$f" = true ]; then
        echo -e "\033[0;31mForcing deletion of containers!\033[0m"
        docker compose -f ./docker-compose.local.test.yml down --rmi all -v --remove-orphans
    else
        echo -e "\033[0;31mRunning docker compose down!\033[0m"
        docker compose -f ./docker-compose.local.test.yml down
    fi
fi

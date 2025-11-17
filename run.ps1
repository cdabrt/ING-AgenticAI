#!/bin/bash
docker compose -f docker-compose.local.yml down --rmi all -v --remove-orphans
docker compose -f docker-compose.local.yml up -d
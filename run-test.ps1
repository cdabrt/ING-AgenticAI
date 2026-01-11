#!/bin/bash
docker compose -f docker-compose.local.test.yml down --rmi all -v --remove-orphans
docker compose -f docker-compose.local.test.yml up -d

#!/bin/bash
docker compose -f docker-compose.local.yml down 
docker compose -f docker-compose.local.yml up -d
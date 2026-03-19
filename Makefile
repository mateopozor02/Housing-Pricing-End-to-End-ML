.PHONY: help build build-backend build-frontend up down logs logs-backend logs-frontend shell-backend shell-frontend health clean prune

# ============================================================================
# Makefile for Docker Development
# Usage: make [target]
# ============================================================================

# Default target
.DEFAULT_GOAL := help

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

# Variables
COMPOSE_FILE := docker-compose.yml
BACKEND_IMAGE := housing-api:latest
FRONTEND_IMAGE := housing-ui:latest

# ============================================================================
# Help
# ============================================================================
help: ## Display this help message
	@echo "$(BLUE)Housing Price Prediction - Docker Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Available Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make build           # Build all Docker images"
	@echo "  make up              # Start all services"
	@echo "  make down            # Stop all services"
	@echo "  make logs            # View logs for all services"
	@echo "  make health          # Check health of all services"

# ============================================================================
# Build Targets
# ============================================================================
build: ## Build all Docker images (backend and frontend)
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build --progress=plain

build-backend: ## Build backend image only
	@echo "$(BLUE)Building backend image...$(NC)"
	docker build -f Dockerfile.backend -t $(BACKEND_IMAGE) .
	@echo "$(GREEN)✓ Backend image built successfully$(NC)"

build-frontend: ## Build frontend image only
	@echo "$(BLUE)Building frontend image...$(NC)"
	docker build -f Dockerfile.frontend -t $(FRONTEND_IMAGE) .
	@echo "$(GREEN)✓ Frontend image built successfully$(NC)"

# ============================================================================
# Service Management
# ============================================================================
up: ## Start all services (detached mode)
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo ""
	@echo "$(BLUE)Service URLs:$(NC)"
	@echo "  Frontend: http://localhost:8501"
	@echo "  Backend:  http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

up-build: ## Build and start all services
	@echo "$(BLUE)Building and starting services...$(NC)"
	docker-compose up -d --build
	@echo "$(GREEN)✓ Services built and started$(NC)"

down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

restart: ## Restart all services
	@echo "$(BLUE)Restarting services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✓ Services restarted$(NC)"

# ============================================================================
# Logging and Monitoring
# ============================================================================
logs: ## View logs for all services
	docker-compose logs -f

logs-backend: ## View logs for backend service only
	docker-compose logs -f backend

logs-frontend: ## View logs for frontend service only
	docker-compose logs -f frontend

logs-tail: ## View last 50 lines of logs (non-blocking)
	docker-compose logs --tail=50

stats: ## Display live container resource usage
	docker stats

health: ## Check health status of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@docker exec housing-api curl -s -o /dev/null -w "Backend (HTTP %{http_code})\n" http://localhost:8000/health
	@docker exec housing-ui curl -s -o /dev/null -w "Frontend (HTTP %{http_code})\n" http://localhost:8501/_stcore/health
	@echo "$(GREEN)✓ Health check complete$(NC)"

# ============================================================================
# Container Inspection and Debugging
# ============================================================================
ps: ## List running containers
	docker-compose ps

shell-backend: ## Open interactive shell in backend container
	docker exec -it housing-api /bin/bash

shell-frontend: ## Open interactive shell in frontend container
	docker exec -it housing-ui /bin/bash

shell-backend-python: ## Open Python REPL in backend container
	docker exec -it housing-api python

inspect-backend: ## Inspect backend image and container details
	@echo "$(BLUE)=== Backend Image ===$(NC)"
	docker image inspect $(BACKEND_IMAGE)
	@echo ""
	@echo "$(BLUE)=== Backend Container ===$(NC)"
	docker inspect housing-api 2>/dev/null || echo "Container not running"

inspect-frontend: ## Inspect frontend image and container details
	@echo "$(BLUE)=== Frontend Image ===$(NC)"
	docker image inspect $(FRONTEND_IMAGE)
	@echo ""
	@echo "$(BLUE)=== Frontend Container ===$(NC)"
	docker inspect housing-ui 2>/dev/null || echo "Container not running"

history-backend: ## Show image build history for backend
	docker history $(BACKEND_IMAGE)

history-frontend: ## Show image build history for frontend
	docker history $(FRONTEND_IMAGE)

# ============================================================================
# Testing and Validation
# ============================================================================
test-api: ## Test API health endpoint
	@echo "$(BLUE)Testing API health endpoint...$(NC)"
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@echo "$(GREEN)✓ API is responding$(NC)"

test-api-docs: ## Open API documentation in browser
	@echo "$(BLUE)Opening API documentation...$(NC)"
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || echo "Please visit http://localhost:8000/docs"

test-all: health test-api ## Run all tests

# ============================================================================
# Cleanup
# ============================================================================
down-volumes: ## Stop services and remove volumes
	@echo "$(BLUE)Stopping services and removing volumes...$(NC)"
	docker-compose down -v
	@echo "$(GREEN)✓ Services and volumes removed$(NC)"

clean: ## Remove containers, images, and volumes (⚠️  WARNING: DESTRUCTIVE)
	@echo "$(RED)⚠️  WARNING: This will remove all containers and images$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v --rmi all; \
		echo "$(GREEN)✓ Cleanup complete$(NC)"; \
	else \
		echo "$(BLUE)Cleanup cancelled$(NC)"; \
	fi

prune: ## Remove unused Docker resources
	@echo "$(BLUE)Pruning unused Docker resources...$(NC)"
	docker system prune -f
	@echo "$(GREEN)✓ Prune complete$(NC)"

prune-all: ## Remove all unused Docker resources including images (⚠️  WARNING)
	@echo "$(RED)⚠️  WARNING: This will remove all unused Docker resources$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker system prune -a -f; \
		echo "$(GREEN)✓ Full prune complete$(NC)"; \
	else \
		echo "$(BLUE)Prune cancelled$(NC)"; \
	fi

# ============================================================================
# Development Workflow
# ============================================================================
dev: ## Start services for development with log monitoring
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose up

dev-build: ## Rebuild images and start in development mode
	@echo "$(BLUE)Rebuilding and starting development environment...$(NC)"
	docker-compose up --build

logs-all: ## View logs from all services (alias for logs)
	docker-compose logs -f

rebuild-backend: ## Rebuild backend image and restart
	@echo "$(BLUE)Rebuilding backend...$(NC)"
	docker-compose up -d --build backend
	@echo "$(GREEN)✓ Backend rebuilt and restarted$(NC)"

rebuild-frontend: ## Rebuild frontend image and restart
	@echo "$(BLUE)Rebuilding frontend...$(NC)"
	docker-compose up -d --build frontend
	@echo "$(GREEN)✓ Frontend rebuilt and restarted$(NC)"

copy-env: ## Copy environment template to .env
	@if [ -f .env ]; then \
		echo "$(RED).env already exists$(NC)"; \
	else \
		cp .env.docker.example .env; \
		echo "$(GREEN)✓ .env created from template$(NC)"; \
		echo "$(BLUE)Please edit .env with your configuration$(NC)"; \
	fi

# ============================================================================
# Production
# ============================================================================
build-prod: ## Build production images with version tags
	@echo "$(BLUE)Building production images...$(NC)"
	docker build -f Dockerfile.backend -t housing-api:prod .
	docker build -f Dockerfile.frontend -t housing-ui:prod .
	@echo "$(GREEN)✓ Production images built$(NC)"

push-prod: ## Push production images to registry (configure registry first)
	@echo "$(RED)Please configure docker registry before pushing$(NC)"
	@echo "Example: docker tag housing-api:prod myregistry/housing-api:latest"
	@echo "Then run: docker push myregistry/housing-api:latest"

# ============================================================================
# Utilities
# ============================================================================
version: ## Display Docker and Docker Compose versions
	@echo "$(BLUE)Docker Version:$(NC)"
	@docker --version
	@echo "$(BLUE)Docker Compose Version:$(NC)"
	@docker-compose --version

info: ## Display system and configuration information
	@echo "$(BLUE)=== System Information ===$(NC)"
	@docker --version
	@docker-compose --version
	@echo ""
	@echo "$(BLUE)=== Running Containers ===$(NC)"
	@docker-compose ps
	@echo ""
	@echo "$(BLUE)=== Available Images ===$(NC)"
	@docker images | grep -E "housing-|REPOSITORY"
	@echo ""
	@echo "$(BLUE)=== Network ===$(NC)"
	@docker network inspect housing-network 2>/dev/null | grep -A 5 "Containers" || echo "Network not created yet"

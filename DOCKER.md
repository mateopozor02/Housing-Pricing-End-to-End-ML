# Docker Setup Guide

This guide explains how to run the Housing Price Prediction application using Docker.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Building and Running](#building-and-running)
- [Environment Configuration](#environment-configuration)
- [Docker Compose](#docker-compose)
- [Container Management](#container-management)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

```bash
# 1. Clone and navigate to the project
cd /path/to/Housing-Pricing-End-to-End-ML

# 2. Create environment file
cp .env.docker.example .env

# 3. Start both services with docker-compose
docker-compose up -d

# 4. Access services
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Architecture

### Multi-Stage Builds

Both Dockerfiles use multi-stage builds for optimal image size and security:

**Stage 1: Builder**
- Installs build tools and dependencies
- Uses `uv` package manager for fast, reliable dependency installation
- Creates virtual environment with all packages

**Stage 2: Runtime**
- Uses minimal `python:3.10-slim` base image
- Copies only necessary dependencies from builder stage
- Removes build tools to reduce final image size
- Runs as non-root user (`appuser`) for security

### Benefits

- **Smaller Image Size**: ~400MB backend, ~500MB frontend (vs 1GB+ with single stage)
- **Security**: No build tools or unnecessary packages in production image
- **Layer Caching**: Docker caches builder stage, speeds up rebuilds
- **Fast Startup**: Minimal dependencies in final image

## Prerequisites

### Required

- **Docker**: 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: 1.29+ (included with Docker Desktop)
- **Disk Space**: ~2GB for both images
- **.env file**: Copy `.env.docker.example` to `.env`

### Optional

- **AWS Credentials**: For S3 integration (set in `.env`)
- **Pre-trained Model**: `models/lightgbm_best_model.pkl` (or download from S3)

## Building and Running

### Option 1: Using Docker Compose (Recommended for Local Development)

```bash
# Build all services
docker-compose build

# Start services (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Option 2: Building Individual Services

#### Build Backend Only

```bash
docker build -f Dockerfile.backend -t housing-api:latest .
```

#### Build Frontend Only

```bash
docker build -f Dockerfile.frontend -t housing-ui:latest .
```

#### Run Backend Standalone

```bash
docker run -d \
  --name housing-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e S3_BUCKET=housing-pricing-regression-data \
  -e AWS_REGION=us-east-2 \
  housing-api:latest
```

#### Run Frontend Standalone

```bash
docker run -d \
  --name housing-ui \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e API_URL=http://localhost:8000/predict \
  housing-ui:latest
```

## Environment Configuration

### Setting Up .env File

```bash
# Copy template
cp .env.docker.example .env

# Edit with your values
vim .env
```

### Required Variables

```env
# AWS S3 (optional, but recommended for production)
S3_BUCKET=housing-pricing-regression-data
AWS_REGION=us-east-2
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# API Configuration
API_URL=http://backend:8000/predict
```

### Service-Specific Configuration

**Backend Configuration**
- `MLFLOW_TRACKING_URI`: MLFlow tracking server (optional)
- `S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: S3 access

**Frontend Configuration**
- `API_URL`: Backend API endpoint
- `STREAMLIT_SERVER_*`: Streamlit server settings (pre-configured in Dockerfile)

## Docker Compose

### Services

**Backend Service**
- Image: Built from `Dockerfile.backend`
- Port: 8000
- Health Check: Every 30s via `/health` endpoint
- Resource Limits: 2 CPU, 2GB RAM
- Volumes:
  - `./models`: Model storage
  - `./data`: Data storage
- Restart Policy: `unless-stopped`

**Frontend Service**
- Image: Built from `Dockerfile.frontend`
- Port: 8501
- Health Check: Every 30s via Streamlit health endpoint
- Depends On: Backend (waits for healthy status)
- Resource Limits: 1 CPU, 1GB RAM
- Restart Policy: `unless-stopped`

### Network

Services communicate via `housing-network` bridge network:
- Frontend → Backend: `http://backend:8000/predict`
- Both services isolated from host network

## Container Management

### Viewing Running Containers

```bash
docker ps
docker ps -a  # Include stopped containers
```

### Accessing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Last 50 lines
docker-compose logs --tail=50

# Follow output
docker-compose logs -f
```

### Executing Commands in Container

```bash
# Execute command
docker exec -it housing-api bash

# Run Python command
docker exec housing-api python -c "print('test')"

# Interactive bash
docker exec -it housing-api /bin/bash
```

### Checking Container Health

```bash
# View health status
docker ps

# Manual health check
docker exec housing-api curl -f http://localhost:8000/health
docker exec housing-ui curl -f http://localhost:8501/_stcore/health
```

### Restarting Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart backend
docker-compose restart frontend

# Rebuild and restart
docker-compose up -d --build backend
```

## Production Deployment

### Building for Production

```bash
# Build with release tag
docker build -f Dockerfile.backend -t housing-api:v1.0.0 .
docker build -f Dockerfile.frontend -t housing-ui:v1.0.0 .

# Push to registry (example with Docker Hub)
docker tag housing-api:v1.0.0 myregistry/housing-api:v1.0.0
docker push myregistry/housing-api:v1.0.0
```

### Docker Compose Override for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.9'

services:
  backend:
    image: myregistry/housing-api:v1.0.0
    restart: always
    deploy:
      replicas: 3  # Multiple replicas with load balancer
      
  frontend:
    image: myregistry/housing-ui:v1.0.0
    restart: always
```

### Running Production Stack

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

Use Docker images as base for Kubernetes deployment:

```bash
# Tag for deployment
docker tag housing-api:latest gcr.io/your-project/housing-api:latest
docker push gcr.io/your-project/housing-api:latest
```

Then create Kubernetes manifests using the same image.

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :8000
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different ports in docker-compose
BACKEND_PORT=8001 FRONTEND_PORT=8502 docker-compose up -d
```

### Out of Memory

```bash
# Check container memory usage
docker stats

# Increase Docker resource limits in Docker Desktop UI or:
# Edit ~/.docker/daemon.json
{
  "memory": "4g",
  "memswap": "4g"
}
```

### Volumes Not Persisting

```bash
# Use absolute paths
docker-compose down -v  # Remove volumes
docker-compose up -d    # Recreate
```

### API Connection Error in Frontend

```bash
# Verify backend is running
docker-compose logs backend

# Test backend endpoint
docker exec housing-ui curl http://backend:8000/health

# Check network
docker network inspect housing-network
```

### S3 Credentials Not Working

```bash
# Verify credentials in .env
cat .env

# Test AWS credentials
docker exec housing-api aws s3 ls

# Check environment variables in container
docker exec housing-api env | grep AWS
```

### Container Crashes on Startup

```bash
# View detailed logs
docker-compose logs backend
docker-compose logs frontend

# Run interactively for debugging
docker run -it --rm housing-api:latest /bin/bash

# Check image integrity
docker image inspect housing-api:latest
```

## Best Practices

### 1. Security

✅ **Do**
- Run as non-root user (`appuser`)
- Use minimal base images (`python:3.10-slim`)
- Update base images regularly
- Use `.dockerignore` to exclude sensitive files
- Never commit sensitive data in Dockerfile

❌ **Don't**
- Run as root user
- Include credentials in Dockerfile
- Use latest tag without version pinning
- Expose unnecessary ports

### 2. Performance

✅ **Do**
- Use multi-stage builds
- Leverage Docker layer caching
- Pin base image versions (e.g., `python:3.10-slim`)
- Use `.dockerignore` effectively
- Minimize final image size

❌ **Don't**
- Install unnecessary build tools in final stage
- Copy entire directory (be specific)
- Create large layers
- Use inefficient base images

### 3. Reliability

✅ **Do**
- Implement health checks
- Use restart policies
- Set resource limits
- Use volumes for persistence
- Log to stdout/stderr for container logs

❌ **Don't**
- Ignore health check failures
- Log to files inside container
- Assume unlimited resources
- Skip error handling

### 4. Development Workflow

```bash
# Work with local changes
docker-compose down
docker-compose up --build

# Monitor logs while developing
docker-compose logs -f

# Rebuild only changed service
docker-compose up -d --build backend
```

### 5. Maintenance

```bash
# Clean up unused images
docker image prune

# Clean up all unused resources
docker system prune -a

# Check image history
docker history housing-api:latest

# Inspect image metadata
docker image inspect housing-api:latest
```

## Testing

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend (when running)
curl http://localhost:8501/_stcore/health

# Full API test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"feature": "value"}]'
```

### Performance Testing

```bash
# Check container resource usage
docker stats

# Load test (install Apache Bench first)
ab -n 100 -c 10 http://localhost:8000/health
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

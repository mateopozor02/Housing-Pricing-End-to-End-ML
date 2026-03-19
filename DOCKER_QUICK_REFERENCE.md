# Docker Quick Reference

Fast reference guide for common Docker commands in this project.

## 🚀 Quick Start

```bash
# Copy and configure environment
cp .env.docker.example .env
vim .env  # Add your AWS credentials

# Build and start everything
make up-build

# View services running
make ps

# Check health
make health
```

## 🐳 Standard Commands

| Command | Description |
|---------|-------------|
| `make up` | Start services |
| `make down` | Stop services |
| `make logs` | View all logs (follow mode) |
| `make health` | Check service health |
| `make ps` | List running containers |
| `make rebuild-backend` | Rebuild and restart backend |
| `make rebuild-frontend` | Rebuild and restart frontend |

## 🔧 Building Images

```bash
# Build all images
make build

# Build specific service
make build-backend
make build-frontend

# Build for production
make build-prod
```

## 📊 Monitoring

```bash
# View logs (follow)
make logs

# View specific service logs
make logs-backend
make logs-frontend

# View last 50 lines without following
make logs-tail

# Monitor resource usage
make stats

# Check service health
make health
```

## 🔍 Debugging

```bash
# Open shell in backend
make shell-backend

# Open shell in frontend
make shell-frontend

# Open Python REPL in backend
make shell-backend-python

# Inspect backend image/container
make inspect-backend
make inspect-frontend

# View image build history
make history-backend
make history-frontend

# Test API endpoint
make test-api

# Open API documentation
make test-api-docs
```

## 🗑️ Cleanup

```bash
# Stop and remove volumes
make down-volumes

# Clean up unused resources
make prune

# Full cleanup (destructive - removes all images)
make clean

# All unused resources including images
make prune-all
```

## 📋 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:8501 | Streamlit UI |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| API Health | http://localhost:8000/health | Health check endpoint |

## ⚙️ Environment Configuration

Create `.env` file with:

```env
# Required for S3 integration
S3_BUCKET=housing-pricing-regression-data
AWS_REGION=us-east-2
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# API URL
API_URL=http://backend:8000/predict
```

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# API test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"feature": "value"}]'
```

## 🐍 Development Workflow

```bash
# Start for development (logs visible)
make dev

# Rebuild and restart specific service after code changes
make rebuild-backend
make rebuild-frontend

# Open shell to test code
make shell-backend-python
```

## 📌 Common Issues

### Port Already in Use
```bash
# Kill process on port 8000
lsof -i :8000 | grep -v COMMAND | awk '{print $2}' | xargs kill -9

# Or use different port
# Edit docker-compose.yml and change port mapping
```

### Container Crashes
```bash
# Check logs
make logs-backend
make logs-frontend

# Inspect container
docker-compose logs backend
```

### S3 Connection Error
```bash
# Verify credentials
cat .env

# Test AWS access
docker exec housing-api aws s3 ls
```

### Frontend Can't Connect to Backend
```bash
# Check IP networking
docker network inspect housing-network

# Test from frontend container
docker exec housing-ui curl http://backend:8000/health
```

## 📖 More Information

- See [DOCKER.md](DOCKER.md) for comprehensive guide
- See [AGENTS.md](AGENTS.md) for project architecture
- See [README.md](README.md) for general project info

## 🎯 Typical Day-to-Day Commands

```bash
# Start work
make up
make logs

# Check everything is healthy
make health

# Make code changes, rebuild a service
make rebuild-backend

# Test API
make test-api

# Debug in container
make shell-backend-python
  >>> import src.api.main
  >>> # test code

# Before committing
make prune

# End of day
make down
```

## 🔐 Security Reminders

- ✅ Never commit `.env` file with secrets
- ✅ Use `.env.docker.example` as template
- ✅ Containers run as non-root user `appuser`
- ✅ Keep base images updated
- ✅ Review `.dockerignore` to exclude sensitive files

## 📚 Docker Command Cheatsheet

```bash
# Images
docker images                          # List images
docker build -t name:tag .             # Build image
docker rmi image_id                    # Remove image
docker image prune                     # Remove unused images

# Containers
docker ps                              # List running
docker ps -a                           # List all
docker run -d --name name image:tag    # Run detached
docker exec -it name bash              # Execute command
docker logs -f name                    # View logs
docker stop name                       # Stop container
docker rm name                         # Remove container

# Docker Compose
docker-compose up -d                   # Start all services
docker-compose down                    # Stop all services
docker-compose logs -f                 # View logs
docker-compose ps                      # List services
docker-compose exec service bash       # Access container

# Cleanup
docker system prune                    # Remove unused
docker system prune -a                 # Remove all unused
```

---

**Last Updated**: February 2026  
**Python Version**: 3.10  
**Base Image**: python:3.10-slim

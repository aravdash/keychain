# Multi-stage build for Next.js frontend and Python backend

# Frontend build stage
FROM node:18-alpine AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Python backend stage
FROM python:3.9-slim AS backend
WORKDIR /app/python-backend

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY python-backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python backend code
COPY python-backend/ .

# Final stage
FROM node:18-alpine AS runtime
WORKDIR /app

# Install Python in the Node.js container
RUN apk add --no-cache python3 py3-pip

# Copy built frontend
COPY --from=frontend-builder /app/.next ./.next
COPY --from=frontend-builder /app/public ./public
COPY --from=frontend-builder /app/package*.json ./
COPY --from=frontend-builder /app/node_modules ./node_modules

# Copy Python backend
COPY --from=backend /app/python-backend ./python-backend
COPY --from=backend /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Expose ports
EXPOSE 3000 5000

# Create startup script
RUN echo '#!/bin/sh\n\
cd /app/python-backend && python app.py &\n\
cd /app && npm start\n\
wait' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
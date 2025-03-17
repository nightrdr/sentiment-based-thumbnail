# Stage 1: Build the application
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps

# Copy the rest of the application code and build the Next.js app
COPY . .
RUN npm run build

# Stage 2: Run the application
FROM node:22-alpine AS runner

WORKDIR /app
ENV NODE_ENV production

# Copy the built application from the builder stage
COPY --from=builder /app ./

# Expose the port that the Next.js app runs on
EXPOSE 3000

# Start the Next.js app
CMD ["npm", "run", "dev"]

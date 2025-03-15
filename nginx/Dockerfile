FROM nginx:latest

# Remove the default configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy your custom configuration file into the container.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80.
EXPOSE 80

# Frontend (Vite + TanStack Start). Runs the Vite dev server — the production
# build targets Cloudflare (not a Node server), so dev mode is the portable way
# to containerize it for a local demo.
FROM node:20-slim

WORKDIR /app

# npm install (not ci): this project's lockfile drifts from package.json for
# some transitive deps; install resolves them rather than failing the build.
COPY package*.json ./
RUN npm install

COPY . .

# Baked into the client bundle; the browser (on the host) reaches Express here.
ENV VITE_API_URL=http://localhost:8000

EXPOSE 8080
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "8080"]

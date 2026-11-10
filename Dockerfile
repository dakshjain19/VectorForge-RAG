# Stage 1: Build
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN g++ -std=c++17 -O2 main.cpp -o db -lpthread

# Stage 2: Run
FROM ubuntu:22.04 AS runner

WORKDIR /app

COPY --from=builder /app/db .
COPY --from=builder /app/index.html .

EXPOSE 8080

CMD ["./db"]
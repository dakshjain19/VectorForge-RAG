FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN g++ -std=c++17 -O2 -Wall -Wextra main.cpp -o db -pthread

FROM ubuntu:22.04 AS runner
WORKDIR /app
COPY --from=builder /app/db .
COPY --from=builder /app/index.html .
RUN mkdir -p /app/data
ENV SERVER_HOST=0.0.0.0
EXPOSE 8080
CMD ["./db"]
